"""LUMEFBPICModel(ActionModel) wrapping a lume-fbpic simulator."""

from __future__ import annotations

from beamphysics import ParticleGroup
from lume.actions import Action, ActionModel, ReadOnlyActionMixin
from lume.staged_model import FinalParticlesMixIn
from lume.variables import ScalarVariable
from lume_fbpic.simulator import BaseSimulator
from pathlib import Path
import h5py
import inspect
import lume_fbpic.actions
import math
import typing


class LUMEFBPICModel(FinalParticlesMixIn, ActionModel[BaseSimulator]):
    """LUMEModel using the actions framework, wrapping a `BaseSimulator`
    (`FBPICSimulator`, or another subclass).

    No `InitialParticlesMixIn` -- the plasma-accelerator stage makes its own bunch (by injection
    off a density downramp or dopant, or from a driver bunch's wake), it doesn't consume an
    externally-supplied one. It is the first stage of a `StagedModel` chain.

    Like `LUMECheetahModel` and `LUMEImpactModel`, `set()` applies the new parameter values and
    then runs the simulation, so `get()` afterwards returns outputs for those inputs. A real
    fbpic run costs minutes to hours, so pass `dummy_run=True` (as `LUMEImpactModel` allows) to
    make `set()` only update parameters without running -- useful for building up a
    configuration, or in tests.

    `archive()` bundles the simulator's config with the action definitions and the values of the
    read-only scalar actions (the outputs); the inputs are in the config. `from_archive()`
    rebuilds the model, actions included. For an archive written without the final particles, the
    recorded output values are put in the simulator's `stats`, where the output actions find
    them; setting an input makes them stale, and they read NaN until `reset()`.

    `reset()` restores the simulator's starting state (for a model loaded from an archive, the
    archived config and results); it never runs the simulation.

    An action made without a `unit` gets its default one: see `lume_fbpic.actions`.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        actions: list[Action],
        dummy_run: bool = False,
    ) -> None:
        _set_default_units(simulator, actions)
        super().__init__(simulator=simulator, action_variables=actions)
        self.dummy_run = dummy_run

    def archive(
        self,
        h5=None,
        *,
        save_final_particles: bool = False,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
    ):
        """Archive the simulator config and the actions into one HDF5 file.

        Written on top of `BaseSimulator.archive()` (the simulator's config, and with
        `save_final_particles=True` the final `ParticleGroup` and `stats`), plus an `actions`
        group with, for each action, its class and parameters and, for a read-only scalar action,
        its current value (NaN where there is no result yet). The inputs are not stored as
        values; they are in the config, which is the current one.

        Parameters
        ----------
        h5 : str, Path, or h5py.File, optional
            Destination. If None, a fingerprint-based filename is used.
        save_final_particles : bool, optional
            Also store the final particles and `stats`; see `BaseSimulator.archive()`.
        input_dirs : str, Path or list of them, optional
            Where the input files are; see `BaseSimulator.archive()`.

        Returns
        -------
        The h5 argument, or the generated filename if it was None.
        """
        h5 = self.simulator.archive(
            h5, save_final_particles=save_final_particles, input_dirs=input_dirs
        )  # names the file
        if isinstance(h5, (str, Path)):
            with h5py.File(h5, "a") as g:
                self._archive_actions(g)
        else:
            self._archive_actions(h5)
        return h5

    @property
    def final_particles(self) -> ParticleGroup | None:
        """The last run's final electron bunch (`BaseSimulator.final_particles`), or None."""
        return self.simulator.final_particles

    @classmethod
    def from_archive(
        cls,
        h5,
        *,
        dummy_run: bool = False,
        working_directory: str | Path | None = None,
        input_dirs: str | Path | typing.Sequence[str | Path] | None = None,
    ) -> "LUMEFBPICModel":
        """Rebuild a model, actions included, from a file written by `archive()`.

        The simulator is restored with `BaseSimulator.from_archive()` (so the final particles
        and `stats` are back if they were saved), and each action is rebuilt from its stored
        class and parameters; the class must be one defined in `lume_fbpic.actions`. `input_dirs`
        is where the simulator looks for the input files the archive refers to; see
        `BaseSimulator.from_archive()`.

        Raises:
            ValueError: If the archive has no `actions` group (it was written by
                `BaseSimulator.archive()`), or an action class is unknown.
        """
        if isinstance(h5, (str, Path)):
            with h5py.File(h5, "r") as g:
                return cls.from_archive(
                    g,
                    dummy_run=dummy_run,
                    working_directory=working_directory,
                    input_dirs=input_dirs,
                )
        if "actions" not in h5:
            raise ValueError(
                "The archive has no actions; it was written by BaseSimulator.archive(). "
                "Use LUMEFBPICModel(simulator, actions) with an action list."
            )

        g = h5["actions"]
        return cls(
            BaseSimulator.from_archive(
                h5,
                working_directory=working_directory,
                input_dirs=input_dirs,
                stats={
                    name: value
                    for name, value in _read_output_values(h5).items()
                    if not math.isnan(value)
                },
            ),
            [
                _action_from_config(
                    {
                        "class": str(g[key].attrs["class"]),
                        "parameters": {
                            name: value.item() if hasattr(value, "item") else value
                            for name, value in g[key]["parameters"].attrs.items()
                        },
                    }
                )
                for key in sorted(g)
            ],
            dummy_run=dummy_run,
        )

    def reset(self) -> None:
        """Restore the simulator's starting state: its config and results when it was built or,
        for a model loaded from an archive, the archived ones (see `BaseSimulator.reset()`).

        Replaces `ActionModel.reset()`, which sets every writable action to its
        `default_value` -- `None` for all of ours, which fails. Nothing is run.
        """
        self.simulator.reset()

    def _archive_action(self, entry, action: Action) -> None:
        """Write one action to `entry`, a group of an open archive: its name and class, its
        parameters as the attributes of a `parameters` group (those that are set), and, for a
        read-only scalar action, its current value as `value` (NaN where there is no result).

        Raises:
            TypeError: If a parameter is not a bool, int, float or string.
        """
        config = _action_to_config(action)
        entry.attrs["name"] = action.name
        entry.attrs["class"] = config["class"]
        parameters = entry.create_group("parameters")
        for key, value in config["parameters"].items():
            if (
                value is None or key == "variable_class"
            ):  # unset, or the class name again
                continue
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError(
                    f"parameter {key!r} of action {action.name!r} is a "
                    f"{type(value).__name__}, which an archive cannot store"
                )
            parameters.attrs[key] = value
        if isinstance(action, ReadOnlyActionMixin) and isinstance(
            action, ScalarVariable
        ):
            entry.attrs["value"] = float(self.get([action.name])[action.name])

    def _archive_actions(self, g) -> None:
        """Write the `actions` group to the open archive `g`: one numbered entry per action, in
        registration order. The inputs are not stored as values; they are in the config.
        """
        actions_group = g.create_group("actions")
        for index, action in enumerate(self.supported_variables.values()):
            self._archive_action(actions_group.create_group(f"{index:04d}"), action)

    def _set(self, values: dict[str, typing.Any]) -> None:
        super()._set(values)
        if values and self.simulator.final_particles is None:
            self.simulator.stats = {}  # the recorded outputs belonged to the old inputs
        if self.dummy_run:
            return
        self.simulator.run()


def _action_from_config(config: dict[str, typing.Any]) -> Action:
    """Rebuild an action from `_action_to_config()`'s output; its class must be one of the
    classes of `lume_fbpic.actions`.

    Raises:
        ValueError: If the class name is not known.
    """
    classes = {
        name: cls
        for name, cls in inspect.getmembers(lume_fbpic.actions, inspect.isclass)
        if issubclass(cls, Action) and cls.__module__ == lume_fbpic.actions.__name__
    }
    name = config["class"]
    if name not in classes:
        raise ValueError(f"Unknown action class {name!r}; known: {sorted(classes)}")
    return classes[name](**config["parameters"])


def _action_to_config(action: Action) -> dict[str, typing.Any]:
    """JSON-compatible description of an action: `{"class": <class name>, "parameters": {...}}`."""
    return {
        "class": type(action).__name__,
        "parameters": action.model_dump(mode="json"),
    }


def _read_output_values(h5) -> dict[str, float]:
    """Read the recorded output values, by action name, from an open file written by
    `LUMEFBPICModel.archive()`; read-only actions without a scalar value (the final-particles
    action) are left out."""
    group = h5["actions"]
    return {
        str(group[key].attrs["name"]): float(group[key].attrs["value"])
        for key in sorted(group)
        if "value" in group[key].attrs
    }


def _set_default_units(simulator: BaseSimulator, actions: list[Action]) -> None:
    """Give each action that was made without a `unit` the one its `default_unit()` finds.

    An action with a `unit` passed to it, `None` included, keeps it.
    """
    for action in actions:
        default_unit = getattr(action, "default_unit", None)
        if default_unit is not None and "unit" not in action.model_fields_set:
            unit = default_unit(simulator)
            if unit is not None:
                action.unit = unit
