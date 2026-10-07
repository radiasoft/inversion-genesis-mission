"""HTU transport lattice as a Cheetah Segment.

Geometry ported from BeamTransportDigitalTwin/examples/htu/htu_lattice.py
(the refined second-generation lattice). Every element that supports it uses
tracking_method='drift_kick_drift' so per-particle energy dependence is exact
— no single-reference-energy linear optics anywhere in the line.

Machine-unit knobs (EMQ currents, chicane R56, kicker currents) are applied
at build time here and at run time by the action variables in htu/model.py,
both through htu.calibrations.
"""

import torch
from cheetah.accelerator import (
    Aperture,
    Dipole,
    Drift,
    HorizontalCorrector,
    Quadrupole,
    Segment,
    VerticalCorrector,
)

from htu import calibrations as cal
from htu.magspec import DutchCalibration, DutchMagSpec
from htu.screens import ScreenGeometry, make_screen

DKD = {"tracking_method": "drift_kick_drift"}
E0 = cal.REFERENCE_ENERGY_EV

# Screens that exist on the line; active ones render images at run time.
SCREEN_NAMES = [
    "PlasmaExit",
    "TCPhosphor",
    "ChicaneSlit",
    "DCPhosphor",
    "Phosphor1",
    "UC_ALineEbeam1",
    "UC_ALineEBeam2",
    "UC_ALineEBeam3",
    *[f"UC_VisaEBeam{i}" for i in range(1, 9)],
]


def _drift(name: str, length: float) -> Drift:
    return Drift(name=name, length=torch.tensor(length), **DKD)


def _quad_from_field(name, length, bore_radius, peak_field, num_steps=10) -> list:
    """PMQ/VISA quad from peak field at bore radius (fixed magnets).

    The bore is a real physical aperture — clip at entrance and exit.
    """
    grad = cal.QUAD_POLARITY * cal.pmq_gradient(bore_radius, peak_field)

    def ap(suffix):
        return Aperture(
            name=f"{name}_AP{suffix}",
            x_max=torch.tensor(bore_radius),
            y_max=torch.tensor(bore_radius),
            shape="elliptical",
        )

    return [
        ap("i"),
        Quadrupole(
            name=name,
            length=torch.tensor(length),
            k1=torch.tensor(cal.gradient_to_k1(grad, E0)),
            num_steps=num_steps,
            metadata={"fixed_gradient_tm": grad},
            **DKD,
        ),
        ap("o"),
    ]


def _quad_from_current(name, length, current, design) -> Quadrupole:
    """EMQ from coil current via the measured calibration."""
    grad = cal.QUAD_POLARITY * cal.emq_current_to_gradient(current, design)
    return Quadrupole(
        name=name,
        length=torch.tensor(length),
        k1=torch.tensor(cal.gradient_to_k1(grad, E0)),
        num_steps=10,
        **DKD,
    )


PIPE_RADIUS_M = 0.010  # 1" OD tube, ~20 mm inner bore

# Per-camera geometry overrides (the rest use screens.DEFAULT_GEOMETRY,
# 7um/1024^2 placeholder until real ECS-dump calibrations are loaded).
# The slit camera needs a wider FOV: in lab frame the beam sits ~6 mm off
# the straight axis at R56=200um.
DEFAULT_SCREEN_GEOMETRY_OVERRIDES = {
    "ChicaneSlit": ScreenGeometry(size_x=1024, size_y=1024, calibration_m=20e-6),
}


def camera_geometry(name: str) -> ScreenGeometry:
    """The geometry a camera is built with (for display axis labeling)."""
    from htu.screens import DEFAULT_GEOMETRY

    return DEFAULT_SCREEN_GEOMETRY_OVERRIDES.get(name, DEFAULT_GEOMETRY)

# PMQ triplet rail/hexapod: base drift lengths at hexapod zero. The triplet
# (PMQ1V + L1 + PMQ2H + L2 + PMQ3V) moves as a rigid unit; Z motion trades
# length between these two drifts, X/Y offsets all three quads together.
SRC_TO_PMQ1_M = 0.052  # at Jetz = 6.8 (jet moves upstream for + changes)
PMQTRIP_TO_TCPHOS_M = 0.2158


class TwoJawSlit(Aperture):
    """Two independent horizontal tungsten jaws (mid-chicane energy slit).

    Cheetah's Aperture is symmetric, so this subclass masks each side
    independently: jaw 1 inserts from +x (blocks x > jaw_plus_m), jaw 2
    from -x (blocks x < jaw_minus_m). Crossed jaws = fully closed (zero
    transmission). Kills particles exactly the way Cheetah's Aperture
    does: multiply survival_probabilities by the survived mask.
    """

    def __init__(self, name: str, jaw_plus_m: float = 0.020,
                 jaw_minus_m: float = -0.020):
        super().__init__(name=name)
        self.jaw_plus_m = jaw_plus_m
        self.jaw_minus_m = jaw_minus_m

    @property
    def is_skippable(self) -> bool:
        # never let the Segment fold this into a transfer matrix
        return False

    def track(self, incoming):
        from cheetah.particles import ParticleBeam

        if not isinstance(incoming, ParticleBeam):
            return incoming
        survived_mask = (incoming.x < self.jaw_plus_m) & (
            incoming.x > self.jaw_minus_m
        )
        return ParticleBeam(
            particles=incoming.particles,
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities * survived_mask,
            s=incoming.s,
            species=incoming.species.clone(),
        )


def _pipe_ap(i: int) -> Aperture:
    """One boundary aperture of the EMQ->magspec beam pipe."""
    return Aperture(
        name=f"PIPE_AP{i}",
        x_max=torch.tensor(PIPE_RADIUS_M),
        y_max=torch.tensor(PIPE_RADIUS_M),
        shape="elliptical",
    )


def _kicker(name, max_current, max_field) -> list:
    """Steering magnet as zero-length H+V corrector pair (currents start 0)."""
    md = {"kicker_max_current_a": max_current, "kicker_max_int_field_tm": max_field}
    return [
        HorizontalCorrector(
            name=f"{name}H", length=torch.tensor(0.0), angle=torch.tensor(0.0),
            metadata=dict(md),
        ),
        VerticalCorrector(
            name=f"{name}V", length=torch.tensor(0.0), angle=torch.tensor(0.0),
            metadata=dict(md),
        ),
    ]


def _bend(name: str, length: float, r56_um: float, bend_index: int) -> Dipole:
    """Chicane dipole from the operator R56 setting [um]."""
    sign = -1.0 if bend_index in (1, 4) else 1.0
    angle = sign * cal.chicane_r56_to_angle(r56_um)
    return Dipole(
        name=name, length=torch.tensor(length), angle=torch.tensor(angle),
        metadata={"chicane_sign": sign}, **DKD
    )


def build_htu_segment(
    emq_currents: tuple[float, float, float] = (
        0.683822195172598,
        -0.882403221217054,
        1.085116293768873,
    ),
    chicane_r56_um: float = 200.0,
    screen_geometries: dict[str, ScreenGeometry] | None = None,
    active_screens: tuple[str, ...] = (
        "TCPhosphor",
        "ChicaneSlit",
        "DCPhosphor",
        "Phosphor1",
        "UC_ALineEbeam1",
        "UC_ALineEBeam2",
        "UC_ALineEBeam3",
        *[f"UC_VisaEBeam{i}" for i in range(1, 9)],
    ),
    to_element: str | None = None,  # None = full line incl. undulator FODO
) -> Segment:
    """Build the HTU line from the plasma source.

    to_element: truncate the line after this element. Default None = the
    full line including the four VISA/undulator FODO segments (the
    undulator FIELD itself is not modeled — FODO transport only).
    """
    # The slit camera needs a wider FOV than the 7um/1024 default: in lab
    # frame the beam sits ~6 mm off the straight axis at R56=200um.
    geo = {**DEFAULT_SCREEN_GEOMETRY_OVERRIDES, **(screen_geometries or {})}

    def screen(name):
        return make_screen(
            name, geometry=geo.get(name), is_active=name in active_screens
        )

    # Distance source -> PMQ1 at Jetz = 6.8 (jet moves upstream for positive
    # changes: ~6.8 mm closer / ~15 mm further available).
    elements = [
        screen("PlasmaExit"),
        _drift("SrcToPMQ1", SRC_TO_PMQ1_M),
        # PMQ triplet (fixed permanent magnets: peak field at bore radius)
        *_quad_from_field("PMQ1V", 0.02903, 0.006, -1.242),
        _drift("L1", 0.029035),
        *_quad_from_field("PMQ2H", 0.02890, 0.006, 1.242),
        _drift("L2", 0.0473895),
        *_quad_from_field("PMQ3V", 0.016321, 0.006, -1.107),
        _drift("PMQTripToTCPhos", PMQTRIP_TO_TCPHOS_M),
        screen("TCPhosphor"),
        _drift("TCPhosToChicane", 0.42),
        *_kicker("S1", cal.S_KICKER_MAX_CURRENT_A, cal.S_KICKER_MAX_INT_FIELD_TM),
        # Chicane (operator R56 knob, bends 1/4 sign-flipped)
        _bend("BEND1", 0.175, chicane_r56_um, 1),
        _drift("L12", 0.125),
        _bend("BEND2", 0.175, chicane_r56_um, 2),
        _drift("L23a", 0.15),
        # mid-chicane energy slit: two independent horizontal jaws just
        # upstream of the slit camera (max dispersion -> energy selection)
        TwoJawSlit(name="ChicaneSlitJaws"),
        screen("ChicaneSlit"),
        _drift("L23b", 0.15),
        _bend("BEND3", 0.175, chicane_r56_um, 3),
        _drift("L12b", 0.125),
        _bend("BEND4", 0.175, chicane_r56_um, 4),
        *_kicker("S2", cal.S_KICKER_MAX_CURRENT_A, cal.S_KICKER_MAX_INT_FIELD_TM),
        _drift("DriftToDCPhos", 0.27),
        screen("DCPhosphor"),
        # 1" OD (~20 mm ID) beam pipe runs from ~10 cm upstream of EMQ1
        # straight through to the magspec entrance (per Sam 2026-08-25);
        # approximated by r=10 mm apertures at each element boundary. The
        # magspec's own rectangular vacuum chamber is generous — no
        # entrance aperture.
        _drift("DriftToEMQTrip_a", 0.305),
        _pipe_ap(1),
        _drift("DriftToEMQTrip_b", 0.100),
        # EMQ triplet (electromagnets: measured current calibrations)
        _quad_from_current("EMQ1H", 0.1408, emq_currents[0], "EMQD-113-394"),
        _pipe_ap(2),
        _drift("EMQL1", 0.112735),
        _pipe_ap(3),
        _quad_from_current("EMQ2V", 0.28141, emq_currents[1], "EMQD-113-949"),
        _pipe_ap(4),
        _drift("EMQL2", 0.112735),
        _pipe_ap(5),
        _quad_from_current("EMQ3H", 0.1409, emq_currents[2], "EMQD-113-394"),
        _pipe_ap(6),
        *_kicker("S3", cal.S_KICKER_MAX_CURRENT_A, cal.S_KICKER_MAX_INT_FIELD_TM),
        _drift("DriftToPhos1", 0.084325),
        _pipe_ap(7),
        screen("Phosphor1"),
        _drift("DriftToSpec", 0.28),
        _pipe_ap(8),  # pipe end = magspec entrance
        # Dutch-magnet magspec: drift when off, screen-map element when on
        DutchMagSpec(name="MagSpec", length=0.4826, calibration=DutchCalibration()),
        *_kicker("S4", cal.S_KICKER_MAX_CURRENT_A, cal.S_KICKER_MAX_INT_FIELD_TM),
        _drift("DriftToAline1", 0.4009),
        screen("UC_ALineEbeam1"),
        _drift("DriftToAline2", 0.3825),
        screen("UC_ALineEBeam2"),
        _drift("DriftToAline3", 0.4191),
        screen("UC_ALineEBeam3"),
        _drift("DriftToUndulator", 0.2945),
    ]

    # VISA/undulator FODO sections (VQ pairs as half-quads, VS steering,
    # VISA screens) — included when to_element=None
    for seg_idx in range(1, 5):
        vqf = _quad_from_field(f"VQ{2 * seg_idx - 1}", 0.0504, 0.004, 0.132, num_steps=3)
        vqd = _quad_from_field(f"VQ{2 * seg_idx}", 0.0504, 0.004, -0.132, num_steps=3)
        vd = _drift(f"VD{seg_idx}", 0.018)
        fodo = [*vqd, *vqd, vd, *vqf, *vqf, vd]
        vs_a, vs_b = 2 * seg_idx - 1, 2 * seg_idx
        elements += [
            *fodo,
            *_kicker(
                f"VS{vs_a}", cal.VISA_KICKER_MAX_CURRENT_A, cal.VISA_KICKER_MAX_INT_FIELD_TM
            ),
            screen(f"UC_VisaEBeam{vs_a}"),
            *fodo,
            *fodo,
            *_kicker(
                f"VS{vs_b}", cal.VISA_KICKER_MAX_CURRENT_A, cal.VISA_KICKER_MAX_INT_FIELD_TM
            ),
            screen(f"UC_VisaEBeam{vs_b}"),
            *fodo,
        ]

    if to_element is not None:
        names = [e.name for e in elements]
        if to_element not in names:
            raise ValueError(f"Element {to_element} not found in beamline")
        elements = elements[: names.index(to_element) + 1]

    return Segment(elements)
