"""LPA source parameterization for the HTU twin.

The input phase space is unknown and jitters shot-to-shot, so it is modeled
as explicit, settable parameters — served as PVs like any magnet. Shape:
per-plane Twiss (beta, alpha, normalized emittance) Gaussian plus centroid
and pointing offsets, with rms energy spread; energy-sliced structure
(delta-dependent divergence/pointing) is the known next refinement.
"""

from dataclasses import dataclass

import torch
from cheetah.particles import ParticleBeam

ELECTRON_MASS_MEV = 0.510999


@dataclass
class SourceParams:
    energy_mev: float = 100.0
    energy_spread_pct: float = 5.0  # rms, percent of mean
    charge_pc: float = 30.0
    beta_x_mm: float = 5.0  # Twiss beta at the source waist plane
    beta_y_mm: float = 5.0
    alpha_x: float = 0.0  # MAD sign convention: + = converging (pre-waist)
    alpha_y: float = 0.0
    norm_emit_x_um: float = 1.0  # normalized emittance, mm·mrad (= um)
    norm_emit_y_um: float = 1.0
    x_um: float = 0.0  # centroid offsets
    y_um: float = 0.0
    xp_mrad: float = 0.0  # pointing angles
    yp_mrad: float = 0.0
    num_particles: int = 20_000
    seed: int = 42


def make_beam(p: SourceParams) -> ParticleBeam:
    """Generate the source ParticleBeam (reproducible via the fixed seed).

    Zero inputs are floored to tiny values: an exactly-zero beta/emittance
    makes the 6D covariance singular (and beta divides sigma_px' inside
    Cheetah), so "0" means "cold / point-like", not a crash.

    Normalized -> geometric emittance: eps_geom = eps_n / (beta_rel * gamma),
    with beta_rel ~= 1 (ultra-relativistic; >=20 MeV everywhere here).
    Cheetah's from_twiss centers the beam (mu_* = 0), so centroid/pointing
    offsets are applied by shifting the particle coordinates afterwards
    (columns 0..3 = x, px, y, py; px/py ~ x'/y' for small angles).
    """
    torch.manual_seed(p.seed)
    num_particles = max(int(p.num_particles), 100)  # floor: sane statistics
    gamma = max(p.energy_mev, 1.0) / ELECTRON_MASS_MEV
    beta_x = max(p.beta_x_mm, 1e-3) * 1e-3  # floor: 1 um beta
    beta_y = max(p.beta_y_mm, 1e-3) * 1e-3
    emit_x = max(p.norm_emit_x_um, 1e-6) * 1e-6 / gamma  # floor: 1e-6 mm·mrad
    emit_y = max(p.norm_emit_y_um, 1e-6) * 1e-6 / gamma
    beam = ParticleBeam.from_twiss(
        num_particles=num_particles,
        beta_x=torch.tensor(beta_x),
        alpha_x=torch.tensor(p.alpha_x),
        emittance_x=torch.tensor(emit_x),
        beta_y=torch.tensor(beta_y),
        alpha_y=torch.tensor(p.alpha_y),
        emittance_y=torch.tensor(emit_y),
        energy=torch.tensor(max(p.energy_mev, 1.0) * 1e6),
        sigma_p=torch.tensor(max(p.energy_spread_pct, 1e-4) / 100.0),
        total_charge=torch.tensor(p.charge_pc * 1e-12),
    )
    beam.particles[..., 0] += p.x_um * 1e-6
    beam.particles[..., 1] += p.xp_mrad * 1e-3
    beam.particles[..., 2] += p.y_um * 1e-6
    beam.particles[..., 3] += p.yp_mrad * 1e-3
    return beam
