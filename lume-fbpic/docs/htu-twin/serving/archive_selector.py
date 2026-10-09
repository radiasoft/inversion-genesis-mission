"""Choose which of several archived LPA runs is the active one.

`ArchiveSelector` holds several loaded `LUMEFBPICModel`s -- for example the seven reconstructed
`initial_sample` runs -- and presents the active one as a single LPA model. It adds one writable
enum variable (`LPA_Archive` by default) whose options are the run names; putting an option makes
that run the active one. Everything else it exposes, `get()` and `final_particles` included, is the
active run's, so in a `StagedModel` a switch reaches the downstream stage the same way a new LPA
result would: `StagedModel.set()` sets the LPA stage first and then passes its `final_particles` on.

All the runs must have the same variables (the same action list), so the served names do not change
when the selection does. Variables that set an input (laser energy, ...) go to the active run only.

A run that holds no final particles -- an archive of a reconstructed run -- has none to hand on. With
`synthetic_bunch_particles=N` the selector builds an approximate bunch from that run's recorded moment
descriptor instead, each time `final_particles` is read.
"""

from __future__ import annotations

import math
import typing
import warnings

import numpy
from scipy.constants import c
from lume_fbpic.simulator import ELECTRON_MC2_EV
from scipy.optimize import minimize
from scipy.stats import norm

from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn
from lume.variables import EnumVariable, Variable

from inversion_fbpic.utils.distributions import COORD_NAMES
from lume_fbpic.model import LUMEFBPICModel

from beamphysics import ParticleGroup

DEFAULT_SELECTOR_NAME = "LPA_Archive"


class ArchiveSelector(FinalParticlesMixIn, LUMEModel):
    """The active one of several `LUMEFBPICModel`s, chosen through an enum variable.

    Args:
        models: Run name -> loaded model, in the order the options are listed; the first is the
            active one to start with. Every model must have the same variable names.
        selector_name: Name of the enum variable. It must not be one of the models' variables.
        synthetic_bunch_particles: If given, a run without final particles (an archive of a
            reconstructed run) hands downstream a bunch of this many macroparticles built from
            its recorded moment descriptor. It is built each time `final_particles` is read, and
            is an APPROXIMATION: the descriptor's 33 numbers do not fix a distribution (see
            `particles_from_descriptor`). None means such a run has no particles.

    Raises:
        ValueError: If `models` is empty, the models' variables differ, `selector_name` is
            already a variable name, or a synthetic bunch is asked for and a run without particles
            has no complete recorded descriptor.
    """

    def __init__(
        self,
        models: dict[str, LUMEFBPICModel],
        *,
        selector_name: str = DEFAULT_SELECTOR_NAME,
        synthetic_bunch_particles: int | None = None,
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
        if synthetic_bunch_particles is not None and synthetic_bunch_particles < 2:
            raise ValueError("synthetic_bunch_particles must be at least 2")
        self._models = dict(models)
        self._synthetic_bunch_particles = synthetic_bunch_particles
        self.selector_name = selector_name
        self._active = names[0]
        self._selector = EnumVariable(
            name=selector_name, options=names, default_value=names[0], read_only=False
        )
        if synthetic_bunch_particles is not None:
            for model in self._models.values():  # fail now, not on the first switch
                if model.final_particles is None:
                    self._synthetic_bunch(model)

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
        """The active run's final particles; for a run without them, the bunch built from its
        recorded descriptor if `synthetic_bunch_particles` was given, otherwise None."""
        particles = self.active_model.final_particles
        if particles is None and self._synthetic_bunch_particles is not None:
            return self._synthetic_bunch(self.active_model)
        return particles

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
        return {
            self.selector_name: self._selector,
            **self.active_model.supported_variables,
        }

    def _get(self, names: list[str]) -> dict[str, typing.Any]:
        values = {}
        if self.selector_name in names:
            values[self.selector_name] = self._active
        rest = [name for name in names if name != self.selector_name]
        if rest:
            values.update(self.active_model.get(rest))
        return values

    def _set(self, values: dict[str, typing.Any]) -> None:
        # The enum validated the option already (`LUMEModel.set`), so the switch cannot fail.
        if self.selector_name in values:
            self._active = str(values[self.selector_name])
        rest = {k: v for k, v in values.items() if k != self.selector_name}
        if rest:
            self.active_model.set(rest)

    def _synthetic_bunch(self, model: LUMEFBPICModel) -> ParticleGroup:
        """A bunch built from `model`'s recorded moment descriptor.

        Raises:
            ValueError: If the model has no recorded descriptor, or an incomplete one.
        """
        prefix = "descriptor_"
        names = [name for name in model.supported_variables if name.startswith(prefix)]
        values = model.get(names) if names else {}
        descriptor = {
            name[len(prefix) :]: value
            for name, value in values.items()
            if not math.isnan(value)
        }
        if not descriptor:
            raise ValueError(
                "a run without final particles has no recorded descriptor values"
            )
        return particles_from_descriptor(
            descriptor, n_particles=self._synthetic_bunch_particles
        )


def particles_from_descriptor(
    descriptor: dict[str, float], *, n_particles: int = 20_000, seed: int = 0
) -> ParticleGroup:
    """A synthetic bunch with the moments of a moment descriptor.

    `descriptor` is `inversion_fbpic.utils.distributions.compute_moment_descriptor`'s output in
    its default spline schema (3 momentum centroids, 21 covariance entries, `longitudinal_mean_uz_NN`
    and `longitudinal_rms_uz_NN` for each longitudinal bin, `total_beam_charge_pc`). It is an
    APPROXIMATION of the bunch it summarizes -- those 33 numbers do not determine a distribution:

    - The longitudinal `uz` structure is a mixture of the descriptor's bins. The bins are equal
      in particle count but their means are charge-weighted, so the charge fractions of the bins
      are recovered (nearest to equal) from the descriptor's `mean_uz` and `cov_uz_uz`; `z` is
      Gaussian with the descriptor's `cov_z_z`, cut into bins of those charge fractions, so the
      energy rises along `z` as it does in the descriptor.
    - The transverse coordinates (`x, ux, y, uy`) are Gaussian, conditional on `(z, uz)`, with
      the regression and residual covariance the descriptor's 6D covariance implies.
    - The mean position is zero (the descriptor does not carry it), and the tails of the real
      distribution are not reproduced.

    The result is in the bunch frame (`t = -z / c`, `z` centred on zero), has equal charge weights
    summing to `total_beam_charge_pc` (in coulombs), and is already the selected bunch the descriptor was computed
    on: do not select it again.

    Raises:
        ValueError: If the descriptor lacks a required key.
    """
    coords = COORD_NAMES  # x, ux, y, uy, z, uz
    try:
        covariance = numpy.zeros((6, 6))
        for row in range(6):
            for column in range(row, 6):
                key = f"cov_{coords[row]}_{coords[column]}"
                covariance[row, column] = covariance[column, row] = descriptor[key]
        bins = sorted(k for k in descriptor if k.startswith("longitudinal_mean_uz_"))
        slice_mean = numpy.array([descriptor[k] for k in bins])
        slice_rms = numpy.array([descriptor[k.replace("mean", "rms")] for k in bins])
        mean_ux, mean_uy, mean_uz = (
            descriptor[f"mean_{n}"] for n in ("ux", "uy", "uz")
        )
        charge = descriptor["total_beam_charge_pc"] * 1e-12  # pC -> C
    except KeyError as error:
        raise ValueError(f"descriptor lacks {error.args[0]!r}") from error
    if not bins:
        raise ValueError("descriptor has no longitudinal_mean_uz_NN features")

    rng = numpy.random.default_rng(seed)
    fractions = _slice_charge_fractions(
        slice_mean, slice_rms, mean_uz, covariance[5, 5]
    )
    counts = numpy.floor(fractions * n_particles).astype(int)
    counts[numpy.argmax(fractions)] += n_particles - counts.sum()

    z_sigma = numpy.sqrt(covariance[4, 4])
    edges = numpy.concatenate([[0.0], numpy.cumsum(fractions)])
    z, uz = numpy.empty(n_particles), numpy.empty(n_particles)
    start = 0
    for k, count in enumerate(counts):
        quantile = rng.uniform(
            edges[k], edges[k + 1], count
        )  # a Gaussian slice by charge
        z[start : start + count] = z_sigma * norm.ppf(
            numpy.clip(quantile, 1e-12, 1 - 1e-12)
        )
        uz[start : start + count] = numpy.maximum(
            rng.normal(slice_mean[k], slice_rms[k], count), 1.0
        )
        start += count

    transverse, longitudinal = [0, 1, 2, 3], [4, 5]
    c_tl = covariance[numpy.ix_(transverse, longitudinal)]
    c_ll = covariance[numpy.ix_(longitudinal, longitudinal)]
    regression = c_tl @ numpy.linalg.pinv(c_ll)
    residual = covariance[numpy.ix_(transverse, transverse)] - regression @ c_tl.T
    values, vectors = numpy.linalg.eigh((residual + residual.T) / 2)
    if values.min() < -1e-9 * max(values.max(), 1e-300):
        warnings.warn(
            "the descriptor's 6D covariance is not positive semi-definite; clipped"
        )
    root = vectors * numpy.sqrt(numpy.clip(values, 0.0, None))
    transverse_mean = numpy.array([0.0, mean_ux, 0.0, mean_uy])
    deviation = numpy.stack([z, uz - mean_uz], axis=-1)
    sample = (
        transverse_mean
        + deviation @ regression.T
        + rng.normal(size=(n_particles, 4)) @ root.T
    )
    return ParticleGroup(
        data={
            "x": sample[:, 0],
            "y": sample[:, 2],
            "z": z,
            "px": sample[:, 1] * ELECTRON_MC2_EV,
            "py": sample[:, 3] * ELECTRON_MC2_EV,
            "pz": uz * ELECTRON_MC2_EV,
            "t": -z / c,
            "status": numpy.ones(n_particles, dtype=int),
            "weight": numpy.full(n_particles, charge / n_particles),
            "species": "electron",
        }
    )


def _slice_charge_fractions(
    slice_mean: numpy.ndarray, slice_rms: numpy.ndarray, mean: float, variance: float
) -> numpy.ndarray:
    """Charge fraction of each longitudinal bin: the non-negative fractions, summing to one, that
    reproduce `mean` and `variance` of `uz` as a mixture of the bins and lie nearest to equal.
    Equal fractions, with a warning, if no such fractions exist."""
    bins = len(slice_mean)
    second = slice_rms**2 + slice_mean**2
    result = minimize(
        lambda f: float(numpy.sum((f - 1.0 / bins) ** 2)),
        numpy.full(bins, 1.0 / bins),
        bounds=[(0.0, 1.0)] * bins,
        constraints=[
            {"type": "eq", "fun": lambda f: f.sum() - 1.0},
            {"type": "eq", "fun": lambda f: f @ slice_mean - mean},
            {
                "type": "eq",
                "fun": lambda f: (f @ second - (f @ slice_mean) ** 2) - variance,
            },
        ],
        method="SLSQP",
    )
    if not result.success:
        warnings.warn(
            "the descriptor's longitudinal bins cannot reproduce its mean uz and cov_uz_uz; "
            "using equal charge in every bin"
        )
        return numpy.full(bins, 1.0 / bins)
    fractions = numpy.clip(result.x, 0.0, None)
    return fractions / fractions.sum()
