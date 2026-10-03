"""Choose which of several archived LPA runs is the active one.

`ArchiveSelector` holds several loaded `LUMEFBPICModel`s -- for example the seven reconstructed
`initial_sample` runs -- and presents the active one as a single LPA model. It adds one writable
enum variable (`LPA_Archive` by default) whose options are the run names; putting an option makes
that run the active one. Everything else it exposes, `get()` and `final_particles` included, is the
active run's, so in a `StagedModel` a switch reaches the downstream stage the same way a new LPA
result would: `StagedModel.set()` sets the LPA stage first and then passes its `final_particles` on.

All the runs must have the same variables (the same action list), so the served names do not change
when the selection does. Variables that set an input (laser energy, ...) go to the active run only.
"""

from __future__ import annotations

from typing import Any

from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume.variables import EnumVariable, Variable

from lume_fbpic.model import LUMEFBPICModel

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

DEFAULT_SELECTOR_NAME = "LPA_Archive"


class ArchiveSelector(FinalParticlesMixIn, LUMEModel):
    """The active one of several `LUMEFBPICModel`s, chosen through an enum variable.

    Args:
        models: Run name -> loaded model, in the order the options are listed; the first is the
            active one to start with. Every model must have the same variable names.
        selector_name: Name of the enum variable. It must not be one of the models' variables.

    Raises:
        ValueError: If `models` is empty, the models' variables differ, or `selector_name` is
            already a variable name.
    """

    def __init__(
        self,
        models: dict[str, LUMEFBPICModel],
        *,
        selector_name: str = DEFAULT_SELECTOR_NAME,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("need at least one model to select from")
        names = list(models)
        reference = set(models[names[0]].supported_variables)
        for name in names[1:]:
            other = set(models[name].supported_variables)
            if other != reference:
                difference = sorted(other ^ reference)
                raise ValueError(
                    f"{name!r} has different variables from {names[0]!r}: {difference}"
                )
        if selector_name in reference:
            raise ValueError(f"{selector_name!r} is already the name of a variable")
        self._models = dict(models)
        self.selector_name = selector_name
        self._active = names[0]
        self._selector = EnumVariable(
            name=selector_name, options=names, default_value=names[0], read_only=False
        )

    @property
    def active(self) -> str:
        """The name of the active run."""
        return self._active

    @property
    def active_model(self) -> LUMEFBPICModel:
        """The active run's model."""
        return self._models[self._active]

    @property
    def final_particles(self) -> ParticleGroup | None:
        """The active run's final particles (its synthesized bunch if it has no real ones)."""
        return self.active_model.final_particles

    @property
    def models(self) -> dict[str, LUMEFBPICModel]:
        """The runs, by name."""
        return dict(self._models)

    def reset(self) -> None:
        """Reset every run and make the first one active again."""
        for model in self._models.values():
            model.reset()
        self._active = next(iter(self._models))

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return {self.selector_name: self._selector, **self.active_model.supported_variables}

    def _get(self, names: list[str]) -> dict[str, Any]:
        values = {}
        if self.selector_name in names:
            values[self.selector_name] = self._active
        rest = [name for name in names if name != self.selector_name]
        if rest:
            values.update(self.active_model.get(rest))
        return values

    def _set(self, values: dict[str, Any]) -> None:
        # The enum validated the option already (`LUMEModel.set`), so the switch cannot fail.
        if self.selector_name in values:
            self._active = str(values[self.selector_name])
        rest = {k: v for k, v in values.items() if k != self.selector_name}
        if rest:
            self.active_model.set(rest)
