"""LUMEFBPICModel(ActionModel) wrapping an FBPICSimulator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from lume.actions import Action, ActionModel, ReadOnlyActionMixin, WritableActionMixin
from lume.staged_model import FinalParticlesMixIn
from lume.variables import ScalarVariable

from lume_fbpic.actions import action_from_config, action_to_config, make_actions
from lume_fbpic.handoff import particles_from_descriptor
from lume_fbpic.simulator import FBPICSimulator


class LUMEFBPICModel(FinalParticlesMixIn, ActionModel[FBPICSimulator]):
    """LUMEModel using the actions framework, wrapping an `FBPICSimulator`.

    No `InitialParticlesMixIn` -- the plasma-accelerator stage self-injects (ionization
    injection off a density downramp/dopant), it doesn't consume an externally-supplied
    bunch. Matches the "Plasma Accelerator" (first) stage of the HTUModel staged chain.

    Like `LUMECheetahModel` and `LUMEImpactModel`, `set()` applies the new parameter values and
    then runs the simulation, so `get()` afterwards returns outputs for those inputs. A real
    fbpic run costs minutes to hours, so pass `dummy_run=True` (as `LUMEImpactModel` allows) to
    make `set()` only update parameters without running -- useful for building up a
    configuration, or in tests.

    `archive()` bundles the simulator's config with the action definitions, their input
    values at the last execution and their output values after it; `from_archive()` rebuilds
    the model, actions included. A model loaded from an archive also serves the archived output
    values whenever the live value is NaN -- the case for an archive written without the final
    particles -- until something is `set()`, which makes them stale.

    `reset()` restores the config and results the model had when it was constructed (for a model
    loaded from an archive, the archived ones); it never runs the simulation.

    A model with no final particles -- an archive of a reconstructed run -- can still hand a bunch
    downstream: `synthesize_bunch()` builds an approximate one from the recorded moment
    descriptor, and `final_particles` returns it while the simulator has none. The descriptor
    outputs stay the recorded values; they are not recomputed from the synthetic bunch.
    """

    def __init__(
        self,
        simulator: FBPICSimulator,
        actions: list[Action],
        dummy_run: bool = False,
    ) -> None:
        super().__init__(simulator=simulator, action_variables=actions)
        self.dummy_run = dummy_run
        self._actions = list(actions)  # in the order given; archived in this order
        # Action values recorded around the last execution: (inputs before the run, outputs
        # after it), or None before any run (or if the run did not finish).
        self._executed_inputs: dict[str, float] | None = None
        self._executed_outputs: dict[str, float] | None = None
        self._synthetic_bunch: ParticleGroup | None = None  # see synthesize_bunch()
        # Output values archived with the config, served while the live value is NaN. Set by
        # `from_archive()`; dropped as soon as the model is `set()`.
        self._recorded_outputs: dict[str, float] = {}
        self._initial_recorded_outputs: dict[str, float] = {}
        # The config and results at construction, which `reset()` restores.
        self._initial_state = (
            simulator.hyparams,
            simulator.laser,
            list(simulator.densities),
            simulator.final_particles,
            dict(simulator.stats),
        )

    def archive(
        self,
        h5=None,
        *,
        metadata: dict[str, Any] | None = None,
        outputs: dict[str, float] | None = None,
        save_final_particles: bool = False,
    ):
        """Archive the simulator config, the actions, and their values, into one HDF5 file.

        Written on top of `FBPICSimulator.archive()` (hyparams, laser, densities, and with
        `save_final_particles=True` the final `ParticleGroup` and `stats`), plus an `actions`
        group with, for each action, its class and parameters and -- for scalar actions -- its
        value: the inputs (writable actions) as they were when the last run was executed, and
        the outputs (read-only scalar actions) as they were after it. If nothing has run
        through this model (a `dummy_run` model, or a batch job read back with
        `simulator.load_results()`), the current values are stored and the group attribute
        `executed` is False. `config_changed_since_execution` is True if an input was
        changed after the recorded run, in which case the config stored is the current one and
        no longer the one the recorded values belong to.

        Parameters
        ----------
        h5 : str, Path, or h5py.File, optional
            Destination. If None, a fingerprint-based filename is used.
        metadata : dict, optional
            Free-form notes stored as attributes of a `metadata` group (for example that the
            config was reconstructed, and from what); `read_archive_metadata()` reads them.
        outputs : dict[str, float], optional
            Output values (by action name) recorded from a run executed elsewhere, stored
            instead of reading them from this model; actions not named are stored as NaN. The
            archive is then marked `executed`, with the current inputs. Names must be read-only
            scalar actions.
        save_final_particles : bool, optional
            Also store the final particles and `stats`; see `FBPICSimulator.archive()`.

        Raises
        ------
        ValueError
            If `outputs` names an action that is not a read-only scalar action.

        Returns
        -------
        The h5 argument (or generated filename) passed in.
        """
        if h5 is None:
            h5 = f"lume_fbpic_{self.simulator.fingerprint()}.h5"
        new_h5file = isinstance(h5, (str, Path))
        g = h5py.File(h5, "w") if new_h5file else h5
        try:
            self.simulator.archive(g, save_final_particles=save_final_particles)

            current_inputs = self._input_values()
            if outputs is not None:
                executed = True
                inputs = current_inputs
                known = self._output_values()
                unknown = sorted(set(outputs) - set(known))
                if unknown:
                    raise ValueError(f"outputs names no read-only scalar action: {unknown}")
                recorded = {name: float(outputs.get(name, math.nan)) for name in known}
            else:
                executed = self._executed_inputs is not None
                inputs = self._executed_inputs if executed else current_inputs
                recorded = self._executed_outputs if executed else self._output_values()

            actions_group = g.create_group("actions")
            actions_group.attrs["count"] = len(self._actions)
            actions_group.attrs["executed"] = executed
            actions_group.attrs["config_changed_since_execution"] = (
                executed and inputs != current_inputs
            )
            actions_group.attrs["outputs_supplied"] = outputs is not None
            for index, action in enumerate(self._actions):
                entry = actions_group.create_group(f"{index:04d}")
                config = action_to_config(action)
                entry.attrs["name"] = action.name
                entry.attrs["class"] = config["class"]
                entry.attrs["parameters"] = json.dumps(config["parameters"])
                if action.name in inputs:
                    entry.attrs["role"] = "input"
                    entry.attrs["value"] = inputs[action.name]
                else:
                    entry.attrs["role"] = "output"
                    if action.name in recorded:
                        entry.attrs["value"] = recorded[action.name]
            if metadata:
                metadata_group = g.create_group("metadata")
                for key, value in metadata.items():
                    metadata_group.attrs[key] = (
                        value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
                    )
        finally:
            if new_h5file:
                g.close()
        return h5

    @property
    def final_particles(self) -> ParticleGroup | None:
        """The last run's final electron bunch (`FBPICSimulator.final_particles`), or the bunch
        from `synthesize_bunch()` while the simulator has none; None if neither."""
        particles = self.simulator.final_particles
        return particles if particles is not None else self._synthetic_bunch

    @classmethod
    def from_archive(
        cls,
        h5,
        *,
        action_classes: dict[str, type] | None = None,
        dummy_run: bool = False,
        working_directory: str | Path | None = None,
    ) -> "LUMEFBPICModel":
        """Rebuild a model, actions included, from a file written by `archive()`.

        The simulator is restored with `FBPICSimulator.from_archive()` (so the final particles
        and `stats` are back if they were saved), and each action is rebuilt from its stored
        class and parameters. `action_classes` maps class names to classes for actions defined
        outside `lume_fbpic.actions`. The recorded values are available from
        `read_action_values()`.

        Raises:
            ValueError: If the archive has no `actions` group (it was written by
                `FBPICSimulator.archive()`), or an action class is unknown.
        """
        if isinstance(h5, (str, Path)):
            with h5py.File(h5, "r") as g:
                return cls.from_archive(
                    g,
                    action_classes=action_classes,
                    dummy_run=dummy_run,
                    working_directory=working_directory,
                )
        if "actions" not in h5:
            raise ValueError(
                "The archive has no actions; it was written by FBPICSimulator.archive(). "
                "Use LUMEFBPICModel.from_simulator() or pass an action list."
            )
        simulator = FBPICSimulator.from_archive(h5, working_directory=working_directory)
        actions_group = h5["actions"]
        actions = [
            action_from_config(
                {
                    "class": str(actions_group[key].attrs["class"]),
                    "parameters": json.loads(actions_group[key].attrs["parameters"]),
                },
                action_classes,
            )
            for key in sorted(actions_group)
        ]
        model = cls(simulator, actions, dummy_run=dummy_run)
        recorded = read_action_values(h5)["outputs"]
        model._recorded_outputs = {k: v for k, v in recorded.items() if not math.isnan(v)}
        model._initial_recorded_outputs = dict(model._recorded_outputs)
        return model

    @classmethod
    def from_simulator(cls, simulator: FBPICSimulator, **kwargs) -> "LUMEFBPICModel":
        """Build a model with the default action set for `simulator` (see
        `lume_fbpic.actions.make_actions`)."""
        return cls(simulator, make_actions(simulator), **kwargs)

    def recorded_descriptor(self) -> dict[str, float]:
        """The recorded moment-descriptor outputs, by feature name (the `descriptor_` prefix
        removed): the 33 scalars of an archive made from a dataset or from a run.

        Raises:
            ValueError: If the model has no recorded descriptor values.
        """
        prefix = "descriptor_"
        descriptor = {
            name[len(prefix) :]: value
            for name, value in self._recorded_outputs.items()
            if name.startswith(prefix)
        }
        if not descriptor:
            raise ValueError("the model has no recorded descriptor values")
        return descriptor

    def reset(self) -> None:
        """Restore the config and results the model had when it was constructed.

        Replaces `ActionModel.reset()`, which sets every writable action to its
        `default_value` -- `None` for all of ours, which fails. Nothing is run. For a model
        loaded from an archive this brings back the archived config, results and output values.
        """
        hyparams, laser, densities, final_particles, stats = self._initial_state
        self.simulator.hyparams = hyparams
        self.simulator.laser = laser
        self.simulator.densities = list(densities)
        self.simulator.final_particles = final_particles
        self.simulator.stats = dict(stats)
        self.simulator.finished = False
        self._executed_inputs = self._executed_outputs = None
        self._recorded_outputs = dict(self._initial_recorded_outputs)

    def synthesize_bunch(self, *, n_particles: int = 20_000, seed: int = 0) -> ParticleGroup:
        """Build an approximate bunch from the recorded descriptor and hand it downstream.

        `final_particles` returns it while the simulator has no particles of its own, so a
        `StagedModel` can pass it on. It is synthetic -- the descriptor's 33 numbers do not fix a
        distribution (see `lume_fbpic.handoff.particles_from_descriptor`) -- and is already the
        selected bunch the descriptor describes. It survives `reset()`.

        Raises:
            ValueError: If the recorded descriptor is missing or incomplete.
        """
        self._synthetic_bunch = particles_from_descriptor(
            self.recorded_descriptor(), n_particles=n_particles, seed=seed
        )
        return self._synthetic_bunch

    def _get(self, names: list[str]) -> dict[str, Any]:
        values = super()._get(names)
        for name in names:
            recorded = self._recorded_outputs.get(name)
            value = values.get(name)
            if recorded is not None and isinstance(value, float) and math.isnan(value):
                values[name] = recorded
        return values

    def _input_values(self) -> dict[str, float]:
        """Current value of every writable action, by name."""
        names = [a.name for a in self._actions if isinstance(a, WritableActionMixin)]
        return {name: float(value) for name, value in self.get(names).items()}

    def _output_values(self) -> dict[str, float]:
        """Current value of every read-only scalar action, by name (NaN if no result)."""
        names = [
            a.name
            for a in self._actions
            if isinstance(a, ReadOnlyActionMixin) and isinstance(a, ScalarVariable)
        ]
        return {name: float(value) for name, value in self.get(names).items()}

    def _set(self, values: dict[str, Any]) -> None:
        super()._set(values)
        if values:
            self._recorded_outputs = {}  # the inputs changed; the archived outputs are stale
        if self.dummy_run:
            return
        inputs = self._input_values()
        self.simulator.run()
        if self.simulator.finished:
            self._executed_inputs = inputs
            self._executed_outputs = self._output_values()
        else:  # the simulator did not run (for example it was not configured)
            self._executed_inputs = self._executed_outputs = None


def read_action_values(h5) -> dict[str, Any]:
    """Read the recorded action values from a file written by `LUMEFBPICModel.archive()`.

    Returns `{"executed": bool, "config_changed_since_execution": bool,
    "inputs": {name: value}, "outputs": {name: value}}`; output actions without a scalar value
    (the final-particles action) are left out.
    """
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as g:
            return read_action_values(g)
    group = h5["actions"]
    values: dict[str, dict[str, float]] = {"input": {}, "output": {}}
    for key in sorted(group):
        entry = group[key]
        if "value" in entry.attrs:
            values[str(entry.attrs["role"])][str(entry.attrs["name"])] = float(
                entry.attrs["value"]
            )
    return {
        "executed": bool(group.attrs["executed"]),
        "config_changed_since_execution": bool(group.attrs["config_changed_since_execution"]),
        "inputs": values["input"],
        "outputs": values["output"],
    }


def read_archive_metadata(h5) -> dict[str, Any]:
    """Read the `metadata` attributes written by `LUMEFBPICModel.archive(metadata=...)`; an
    empty dict if there are none."""
    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as g:
            return read_archive_metadata(g)
    if "metadata" not in h5:
        return {}
    return {str(key): value for key, value in h5["metadata"].attrs.items()}
