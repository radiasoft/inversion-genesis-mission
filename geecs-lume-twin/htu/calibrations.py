"""Magnet calibrations and unit conversions for the HTU transport line.

All numbers ported from BeamTransportDigitalTwin/examples/htu/htu_lattice.py
(which refined BLAST-AI-ML/bella_htu_digital_twin). Provenance is noted per
constant; these are measured device calibrations, not tunables.
"""

import math

from scipy.constants import c, e, m_e

REFERENCE_ENERGY_EV = 100e6
# Ported caveat: the chicane angle formula and kicker/corrector angles assume
# a 100 MeV reference. For beams with mean-energy fluctuations the reference
# particle energy must be held constant.


def rigidity(energy_ev: float) -> float:
    """Magnetic rigidity [T*m] at the given total energy [eV]."""
    gamma = energy_ev * e / (m_e * c**2)
    beta = (1 - 1 / gamma**2) ** 0.5
    return m_e * gamma * beta * c / e


# --- EMQ electromagnet quads: measured current -> gradient linear fits ------
# "LBM6_03 Calibration" in EMQD-113-394 / EMQD-113-949 Testing Report-FINAL.pdf
EMQ_CALIBRATIONS = {
    "EMQD-113-394": (2.9217, 0.0965),  # gradient [T/m] = a * I[A] + b
    "EMQD-113-949": (2.9318, 0.0077),
}

# Overall quad polarity convention carried over from the ImpactX lattice
# (k = -1 * gradient there). VALIDATE against measured optics before trusting
# absolute focusing planes.
QUAD_POLARITY = -1.0


def emq_current_to_gradient(current_a: float, design: str) -> float:
    """EMQ coil current [A] -> field gradient [T/m]."""
    a, b = EMQ_CALIBRATIONS[design]
    return a * current_a + b


def emq_gradient_to_current(gradient_tm: float, design: str) -> float:
    """Inverse of emq_current_to_gradient (linear fit is invertible)."""
    a, b = EMQ_CALIBRATIONS[design]
    return (gradient_tm - b) / a


def pmq_gradient(bore_radius_m: float, peak_field_t: float) -> float:
    """PMQ peak field at the bore radius -> gradient [T/m]."""
    return peak_field_t / bore_radius_m


def gradient_to_k1(gradient_tm: float, energy_ev: float) -> float:
    """Field gradient [T/m] -> normalized quad strength k1 [1/m^2]."""
    return gradient_tm / rigidity(energy_ev)


# --- Steering kickers: integrated field calibrations ------------------------
# Integrated field (G.cm -> T.m) and max current from
# HTU_Kickers_SteeringMagnets.pdf; VISA values scaled by 0.5 to account for
# lower peak field (per Sam's recommendation, carried from the source repo).
S_KICKER_MAX_INT_FIELD_TM = 1970e-6
S_KICKER_MAX_CURRENT_A = 5.0
VISA_KICKER_MAX_INT_FIELD_TM = 1970e-6 * 0.5
VISA_KICKER_MAX_CURRENT_A = 5.0


def kicker_current_to_angle(
    current_a: float,
    max_current_a: float,
    max_int_field_tm: float,
    energy_ev: float,
) -> float:
    """Kicker coil current [A] -> deflection angle [rad] at energy_ev."""
    int_field = current_a * max_int_field_tm / max_current_a
    return int_field / rigidity(energy_ev)


def kicker_angle_to_current(
    angle_rad: float,
    max_current_a: float,
    max_int_field_tm: float,
    energy_ev: float,
) -> float:
    """Inverse of kicker_current_to_angle."""
    int_field = angle_rad * rigidity(energy_ev)
    return int_field * max_current_a / max_int_field_tm


# --- Chicane: operator R56 knob [um] -> bend angle --------------------------
# "Updated r56 formula valid near r56=0"; the control-system R56 value does
# not necessarily equal the true R56 if the beam energy differs from 100 MeV.
CHICANE_ANGLE_COEFF = 0.001438389904456


def chicane_r56_to_angle(r56_um: float) -> float:
    """Chicane R56 setting [um] -> single-bend angle [rad] (unsigned)."""
    return CHICANE_ANGLE_COEFF * math.sqrt(max(r56_um, 0.0))


def chicane_angle_to_r56(angle_rad: float) -> float:
    """Inverse of chicane_r56_to_angle."""
    return (angle_rad / CHICANE_ANGLE_COEFF) ** 2


CHICANE_BEND_LENGTH_M = 0.175
CHICANE_L12_M = 0.125  # BEND1 -> BEND2 drift

# --- Chicane dipole B(I): measured field-vs-current scan (Sam, 2026-08-25;
# all four dipoles identical). Nearly linear ~18.1 mT/A with 0.2 mT
# remanence; kept as an interpolation table, not a fit.
CHICANE_BI_CURRENT_A = [0, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
CHICANE_BI_FIELD_T = [v * 1e-3 for v in
                      [0.2, 1.9, 3.5, 8.7, 17.7, 26.7, 35.7, 44.8, 54, 72.5,
                       90.9, 109.2, 127.4, 145.6, 163.6, 181.5, 199.2, 216.9,
                       234.4, 251.9, 269.2, 286.6, 305, 320, 337, 361]]


def chicane_current_to_field(current_a: float) -> float:
    """Chicane dipole coil current [A] -> field [T] (table interpolation)."""
    import numpy as np

    return float(np.interp(current_a, CHICANE_BI_CURRENT_A, CHICANE_BI_FIELD_T))


def chicane_field_to_current(field_t: float) -> float:
    """Inverse of chicane_current_to_field (table is monotonic)."""
    import numpy as np

    return float(np.interp(field_t, CHICANE_BI_FIELD_T, CHICANE_BI_CURRENT_A))


def chicane_field_to_r56_um(field_t: float) -> float:
    """Dipole field [T] -> chicane R56 [um] at the 100 MeV ops convention.

    theta = B*L/Brho(100 MeV); R56 = 2*theta^2*(L12 + 2/3*L_bend) — the
    same closed form the legacy R56 knob inverted.
    """
    theta = field_t * CHICANE_BEND_LENGTH_M / rigidity(REFERENCE_ENERGY_EV)
    return 2.0 * theta**2 * (CHICANE_L12_M + 2.0 * CHICANE_BEND_LENGTH_M / 3.0) * 1e6


def chicane_r56_to_current_a(r56_um: float) -> float:
    """R56 [um] (100 MeV convention) -> coil current [A] via the B-I table."""
    theta = chicane_r56_to_angle(r56_um)
    field = theta * rigidity(REFERENCE_ENERGY_EV) / CHICANE_BEND_LENGTH_M
    return chicane_field_to_current(field)


def chicane_ref_offset_at_slit(bend_angle_rad: float) -> float:
    """Lab-frame offset of the reference orbit at the mid-chicane slit
    plane, relative to the straight (R56 = 0) beamline axis.

    Geometry: sector bend (radius rho = L/theta), drift, opposite bend ->
    parallel displaced path through the slit region. Sign: positive toward
    the low-energy side of the dispersed beam (+x in the local frame, per
    the measured dispersion orientation).
    """
    th = abs(bend_angle_rad)
    if th < 1e-9:
        return 0.0
    rho = CHICANE_BEND_LENGTH_M / th
    return 2.0 * rho * (1.0 - math.cos(th)) + CHICANE_L12_M * math.tan(th)
