"""Prepare a `lume-fbpic` bunch for a downstream beam-line model.

`FBPICSimulator.final_particles` is a lab-frame snapshot: every particle at one instant (`t = 0`)
at its own longitudinal position `z`. A beam-line code wants the opposite -- every particle at one
position, with its own arrival time. `bunch_frame_particles()` makes that change (after selecting
the accelerated bunch), `drift_particles()` moves a bunch ballistically from where it is to a
reference plane, `plasma_exit_z()` names the nominal exit plane of a gas profile,
`beam_moments()` summarizes a bunch as the per-plane Twiss, energy and charge numbers a
parametric source is described by, and `particles_from_descriptor()` builds a SYNTHETIC bunch
from the 33-scalar moment descriptor when the real particles are not available.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.constants import c, e, m_e
from scipy.optimize import minimize
from scipy.stats import norm

from inversion_fbpic.utils.distributions import COORD_NAMES

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

# Electron rest energy [eV]; ParticleGroup momenta are in eV/c.
_MC2_EV = m_e * c**2 / e


def beam_moments(particles: ParticleGroup) -> dict[str, float]:
    """Charge-weighted energy, charge and per-plane Twiss of a bunch.

    Keys: `energy_mev` (mean total energy), `energy_spread_pct` (rms, percent of the mean),
    `charge_pc`, `num_particles`, and for each plane `p` in `x`, `y`: `beta_<p>_mm`, `alpha_<p>`,
    `norm_emit_<p>_um` (normalized emittance in mm mrad), `<p>_um` (centroid) and `<p>p_mrad`
    (mean angle). Angles are `px / p0c`, with `p0c` the mean momentum, as in Cheetah, and
    `alpha` follows the MAD convention (positive when converging). A zero-emittance plane gets
    `beta = alpha = 0`.

    Raises:
        ValueError: If the bunch has fewer than two particles or no charge.
    """
    weights = np.abs(np.asarray(particles.weight, dtype=np.float64))
    if len(weights) < 2 or weights.sum() <= 0:
        raise ValueError("need at least two particles with charge")
    weights = weights / weights.sum()

    def mean(values):
        return float(np.sum(weights * values))

    energy = np.asarray(particles.energy, dtype=np.float64)
    mean_energy = mean(energy)
    p0c = float(np.sqrt(mean_energy**2 - _MC2_EV**2))
    moments = {
        "energy_mev": mean_energy / 1.0e6,
        "energy_spread_pct": 100.0 * float(np.sqrt(mean((energy - mean_energy) ** 2))) / mean_energy,
        "charge_pc": float(np.sum(np.abs(particles.weight))) * 1.0e12,
        "num_particles": float(len(weights)),
    }
    for plane in ("x", "y"):
        position = np.asarray(getattr(particles, plane), dtype=np.float64)
        angle = np.asarray(getattr(particles, f"p{plane}"), dtype=np.float64) / p0c
        centroid, mean_angle = mean(position), mean(angle)
        dx, da = position - centroid, angle - mean_angle
        sxx, sxa, saa = mean(dx * dx), mean(dx * da), mean(da * da)
        emittance = float(np.sqrt(max(sxx * saa - sxa * sxa, 0.0)))
        beta, alpha = (sxx / emittance, -sxa / emittance) if emittance > 0 else (0.0, 0.0)
        moments[f"beta_{plane}_mm"] = beta * 1.0e3
        moments[f"alpha_{plane}"] = alpha
        moments[f"norm_emit_{plane}_um"] = emittance * (p0c / _MC2_EV) * 1.0e6
        moments[f"{plane}_um"] = centroid * 1.0e6
        moments[f"{plane}p_mrad"] = mean_angle * 1.0e3
    return moments


def bunch_frame_particles(
    particles: ParticleGroup,
    *,
    central_fraction: float | None = 0.95,
    uz_min: float | None = 30.0,
) -> ParticleGroup:
    """Select the bunch from a lab-frame snapshot and put it in the bunch frame.

    Selection follows the moment descriptor (`inversion_fbpic.utils.distributions`): keep
    `uz = pz / (m_e c) >= uz_min`, then the central `central_fraction` of the charge-weighted
    distance from the mean on each of the six phase-space axes (`None` skips a step).

    The returned `ParticleGroup` has `t = -(z - <z>) / c`, `<z>` the charge-weighted mean, so that a
    particle ahead of the bunch centre (larger `z`) has a negative arrival time, as in Cheetah's
    `tau = c t`, and `status = 1`. `x, y, z, px, py, pz` and the charge weights are unchanged.

    Raises:
        ValueError: If fewer than two particles remain after the selection.
    """
    x, y, z = (np.asarray(getattr(particles, name), dtype=np.float64) for name in "xyz")
    ux, uy, uz = (
        np.asarray(getattr(particles, f"p{name}"), dtype=np.float64) / _MC2_EV for name in "xyz"
    )
    weight = np.asarray(particles.weight, dtype=np.float64)

    keep = np.ones(len(x), dtype=bool)
    if uz_min is not None:
        keep &= uz >= uz_min
    if central_fraction is not None:
        if not 0 < central_fraction <= 1:
            raise ValueError("central_fraction must be in (0, 1]")
        phase_space = np.stack([x, ux, y, uy, z, uz], axis=-1)[keep]
        if len(phase_space) >= 2:
            centre = np.average(phase_space, axis=0, weights=np.abs(weight[keep]))
            distance = np.abs(phase_space - centre)
            limit = np.quantile(distance, central_fraction, axis=0)
            inside = np.all(distance <= limit, axis=1)
            central = np.zeros(len(x), dtype=bool)
            central[np.flatnonzero(keep)[inside]] = True
            keep = central
    if keep.sum() < 2:
        raise ValueError("fewer than two particles remain after the selection")

    z_mean = np.average(z[keep], weights=np.abs(weight[keep]))
    return ParticleGroup(
        data={
            "x": x[keep],
            "y": y[keep],
            "z": z[keep],
            "px": np.asarray(particles.px, dtype=np.float64)[keep],
            "py": np.asarray(particles.py, dtype=np.float64)[keep],
            "pz": np.asarray(particles.pz, dtype=np.float64)[keep],
            "t": -(z[keep] - z_mean) / c,
            "status": np.ones(int(keep.sum()), dtype=int),
            "weight": weight[keep],
            "species": "electron",
        }
    )


def drift_particles(particles: ParticleGroup, length: float) -> ParticleGroup:
    """Move a bunch ballistically along `z` by `length` metres (negative moves it back).

    Each particle travels in a straight line at its own angle `px / pz`, so `x` and `y` change by
    `length * px / pz` and `z` by `length`. The arrival time `t` changes by the extra time of
    flight relative to the reference particle -- the charge-weighted mean energy `E0`, on axis --
    which is `(length / c) * (E / (pz c) - E0 / p0c)` per particle (`pz c` and `p0c` in eV):
    higher-energy particles gain on the reference when `length` is positive, and the
    charge-weighted mean `t` is not held at zero. This is the same ballistic drift as Cheetah's
    `Drift` with `tracking_method="drift_kick_drift"`; there are no fields, no space charge and no
    scattering. Momenta, charges and `status` are unchanged.
    """
    px, py, pz = (np.asarray(getattr(particles, f"p{n}"), dtype=np.float64) for n in "xyz")
    energy = np.asarray(particles.energy, dtype=np.float64)
    weight = np.abs(np.asarray(particles.weight, dtype=np.float64))
    e0 = float(np.average(energy, weights=weight))
    p0c = float(np.sqrt(e0**2 - _MC2_EV**2))
    return ParticleGroup(
        data={
            "x": np.asarray(particles.x, dtype=np.float64) + length * px / pz,
            "y": np.asarray(particles.y, dtype=np.float64) + length * py / pz,
            "z": np.asarray(particles.z, dtype=np.float64) + length,
            "px": px,
            "py": py,
            "pz": pz,
            "t": np.asarray(particles.t, dtype=np.float64) + (length / c) * (energy / pz - e0 / p0c),
            "status": np.asarray(particles.status),
            "weight": np.asarray(particles.weight, dtype=np.float64),
            "species": particles.species,
        }
    )


def particles_from_descriptor(
    descriptor: dict[str, float], *, n_particles: int = 20_000, seed: int = 0
) -> ParticleGroup:
    """A synthetic bunch with the moments of a moment descriptor.

    `descriptor` is `inversion_fbpic.utils.distributions.compute_moment_descriptor`'s output in
    its default spline schema (3 momentum centroids, 21 covariance entries, `longitudinal_mean_uz_NN`
    and `longitudinal_rms_uz_NN` for each longitudinal bin, `total_beam_charge_c`). It is an
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
    summing to `total_beam_charge_c`, and is already the selected bunch the descriptor was computed
    on: do not select it again.

    Raises:
        ValueError: If the descriptor lacks a required key.
    """
    coords = COORD_NAMES  # x, ux, y, uy, z, uz
    try:
        covariance = np.zeros((6, 6))
        for row in range(6):
            for column in range(row, 6):
                key = f"cov_{coords[row]}_{coords[column]}"
                covariance[row, column] = covariance[column, row] = descriptor[key]
        bins = sorted(k for k in descriptor if k.startswith("longitudinal_mean_uz_"))
        slice_mean = np.array([descriptor[k] for k in bins])
        slice_rms = np.array([descriptor[k.replace("mean", "rms")] for k in bins])
        mean_ux, mean_uy, mean_uz = (descriptor[f"mean_{n}"] for n in ("ux", "uy", "uz"))
        charge = descriptor["total_beam_charge_c"]
    except KeyError as error:
        raise ValueError(f"descriptor lacks {error.args[0]!r}") from error
    if not bins:
        raise ValueError("descriptor has no longitudinal_mean_uz_NN features")

    rng = np.random.default_rng(seed)
    fractions = _slice_charge_fractions(slice_mean, slice_rms, mean_uz, covariance[5, 5])
    counts = np.floor(fractions * n_particles).astype(int)
    counts[np.argmax(fractions)] += n_particles - counts.sum()

    z_sigma = np.sqrt(covariance[4, 4])
    edges = np.concatenate([[0.0], np.cumsum(fractions)])
    z, uz = np.empty(n_particles), np.empty(n_particles)
    start = 0
    for k, count in enumerate(counts):
        quantile = rng.uniform(edges[k], edges[k + 1], count)  # a Gaussian slice by charge
        z[start : start + count] = z_sigma * norm.ppf(np.clip(quantile, 1e-12, 1 - 1e-12))
        uz[start : start + count] = np.maximum(
            rng.normal(slice_mean[k], slice_rms[k], count), 1.0
        )
        start += count

    transverse, longitudinal = [0, 1, 2, 3], [4, 5]
    c_tl = covariance[np.ix_(transverse, longitudinal)]
    c_ll = covariance[np.ix_(longitudinal, longitudinal)]
    regression = c_tl @ np.linalg.pinv(c_ll)
    residual = covariance[np.ix_(transverse, transverse)] - regression @ c_tl.T
    values, vectors = np.linalg.eigh((residual + residual.T) / 2)
    if values.min() < -1e-9 * max(values.max(), 1e-300):
        warnings.warn("the descriptor's 6D covariance is not positive semi-definite; clipped")
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    transverse_mean = np.array([0.0, mean_ux, 0.0, mean_uy])
    deviation = np.stack([z, uz - mean_uz], axis=-1)
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
            "px": sample[:, 1] * _MC2_EV,
            "py": sample[:, 3] * _MC2_EV,
            "pz": uz * _MC2_EV,
            "t": -z / c,
            "status": np.ones(n_particles, dtype=int),
            "weight": np.full(n_particles, charge / n_particles),
            "species": "electron",
        }
    )


def plasma_exit_z(densities) -> float:
    """The nominal plasma-exit plane [m, lab `z`]: the downstream end of the longest density
    extent among `densities` (each profile's `get_z_extent()`).

    That end is a convention of the profile (for `GeneralizedGaussianProfile`, `z0 + 2.8 alpha`),
    not a sharp edge -- the gas density there is not zero -- so pass an explicit plane to
    `TwinStage` when the machine's actual exit is known.
    """
    return float(max(density.get_z_extent()[1] for density in densities))


def _slice_charge_fractions(
    slice_mean: np.ndarray, slice_rms: np.ndarray, mean: float, variance: float
) -> np.ndarray:
    """Charge fraction of each longitudinal bin: the non-negative fractions, summing to one, that
    reproduce `mean` and `variance` of `uz` as a mixture of the bins and lie nearest to equal.
    Equal fractions, with a warning, if no such fractions exist."""
    bins = len(slice_mean)
    second = slice_rms**2 + slice_mean**2
    result = minimize(
        lambda f: float(np.sum((f - 1.0 / bins) ** 2)),
        np.full(bins, 1.0 / bins),
        bounds=[(0.0, 1.0)] * bins,
        constraints=[
            {"type": "eq", "fun": lambda f: f.sum() - 1.0},
            {"type": "eq", "fun": lambda f: f @ slice_mean - mean},
            {"type": "eq", "fun": lambda f: (f @ second - (f @ slice_mean) ** 2) - variance},
        ],
        method="SLSQP",
    )
    if not result.success:
        warnings.warn(
            "the descriptor's longitudinal bins cannot reproduce its mean uz and cov_uz_uz; "
            "using equal charge in every bin"
        )
        return np.full(bins, 1.0 / bins)
    fractions = np.clip(result.x, 0.0, None)
    return fractions / fractions.sum()
