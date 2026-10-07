"""LUME model of the HTU transport line: machine-unit knobs + source PVs.

Design: the authoritative state is (machine_state dict, source_params) held
on the simulator — PVs read/write those, and _apply_machine_state() derives
every element parameter from them at the current beam energy. This keeps
magnet PVs meaning coil amps: change the source energy and the currents hold
while every k1 and bend angle re-derives (the machine doesn't know the beam
energy changed — that's the physics of energy jitter).
"""

import torch
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import BoolVariable, IntVariable
from lume_cheetah import CheetahSimulator, LUMECheetahModel
from lume_cheetah.actions import (
    CheetahReadOnlyNDVariable,
    CheetahReadOnlyScalarVariable,
)
from lume_torch.variables import TorchScalarVariable

from htu import calibrations as cal
from htu.lattice import build_htu_segment
from htu.screens import DEFAULT_GEOMETRY, ScreenGeometry
from htu.source import SourceParams, make_beam

EMQ_DESIGNS = {
    "EMQ1H": "EMQD-113-394",
    "EMQ2V": "EMQD-113-949",
    "EMQ3H": "EMQD-113-394",
}

DEFAULT_MACHINE_STATE = {
    "EMQ1H": 0.683822195172598,
    "EMQ2V": -0.882403221217054,
    "EMQ3H": 1.085116293768873,
    "chicane_current": 2.1688,  # A; gives R56 = 200 um at 100 MeV
    "magspec_on": 0.0,  # off = transport mode (drift-through)
    "magspec_b": 0.825,  # T (Sam's default; live Hall-probe link some other day)
    "pmq_x_mm": 0.0,  # hexapod: triplet transverse offset
    "pmq_y_mm": 0.0,  # hexapod: triplet vertical offset
    "pmq_z_mm": 0.0,  # rail: + = downstream (SrcToPMQ1 grows)
    "pmq1v_tilt_mrad": 0.0,  # per-magnet rotation about the beam axis
    "pmq2h_tilt_mrad": 0.0,
    "pmq3v_tilt_mrad": 0.0,
    "chicane_jaw1_mm": 20.0,  # slit jaw from +x (blocks x > value); retracted
    "chicane_jaw2_mm": -20.0,  # slit jaw from -x (blocks x < value); retracted
    **{f"{s}{p}": 0.0 for s in ["S1", "S2", "S3", "S4"] for p in ["H", "V"]},
    **{f"VS{i}{p}": 0.0 for i in range(1, 9) for p in ["H", "V"]},
}


def _assign(element, attr: str, value) -> None:
    """Assign a scalar or list to a torch buffer, matching dtype/shape."""
    current = getattr(element, attr)
    setattr(element, attr, torch.as_tensor(value, dtype=current.dtype).reshape(current.shape))


def _slit_ref_offset(sim: CheetahSimulator, st: dict) -> float:
    """Lab-frame offset [m] of the reference orbit at the slit plane."""
    field = cal.chicane_current_to_field(st["chicane_current"])
    angle = field * cal.CHICANE_BEND_LENGTH_M / cal.rigidity(
        float(sim.energies["BEND2"])
    )
    return cal.chicane_ref_offset_at_slit(angle)


def _apply_machine_state(sim: CheetahSimulator) -> None:
    """Derive all element parameters from machine_state at current energies."""
    st = sim.machine_state

    # PMQ triplet hexapod/rail: rigid-unit motion. X/Y offset all three
    # quads together; Z trades drift length on either side of the unit.
    # (Known gap: the bore Aperture elements do NOT move with the triplet —
    # off-axis scraping at the bores is slightly underestimated.)
    dx, dy = st["pmq_x_mm"] * 1e-3, st["pmq_y_mm"] * 1e-3
    dz = st["pmq_z_mm"] * 1e-3
    from htu.lattice import PMQTRIP_TO_TCPHOS_M, SRC_TO_PMQ1_M

    # Per-magnet tilt (rotation about the propagation axis) composes with
    # the rigid-unit misalignment inside Cheetah's offset_particle — both
    # are element properties, no manual composition needed.
    for name in ["PMQ1V", "PMQ2H", "PMQ3V"]:
        _assign(getattr(sim.segment, name), "misalignment", [dx, dy])
        _assign(
            getattr(sim.segment, name), "tilt", st[f"{name.lower()}_tilt_mrad"] * 1e-3
        )
    _assign(sim.segment.SrcToPMQ1, "length", SRC_TO_PMQ1_M + dz)
    _assign(sim.segment.PMQTripToTCPhos, "length", PMQTRIP_TO_TCPHOS_M - dz)

    for name, design in EMQ_DESIGNS.items():
        grad = cal.QUAD_POLARITY * cal.emq_current_to_gradient(st[name], design)
        energy = float(sim.energies[name])
        _assign(getattr(sim.segment, name), "k1", cal.gradient_to_k1(grad, energy))

    for elem in sim.segment.elements:
        md = getattr(elem, "metadata", None) or {}
        energy = float(sim.energies[elem.name])
        if "fixed_gradient_tm" in md:  # PMQ / VISA quads: field is fixed
            _assign(elem, "k1", cal.gradient_to_k1(md["fixed_gradient_tm"], energy))
        elif "kicker_max_current_a" in md:
            angle = cal.kicker_current_to_angle(
                st.get(elem.name, 0.0),
                md["kicker_max_current_a"],
                md["kicker_max_int_field_tm"],
                energy,
            )
            _assign(elem, "angle", angle)
        elif hasattr(elem, "jaw_plus_m"):  # the chicane energy slit
            # Jaw PVs are LAB-frame positions relative to the straight
            # (R56=0) axis — the physical tungsten plates are chamber-
            # mounted. Cheetah tracks in the reference-orbit frame, which
            # at the slit sits ref_offset off the straight axis (R56- and
            # energy-dependent), so translate: rel = lab - ref_offset.
            ref_off = _slit_ref_offset(sim, st)
            elem.jaw_plus_m = st["chicane_jaw1_mm"] * 1e-3 - ref_off
            elem.jaw_minus_m = st["chicane_jaw2_mm"] * 1e-3 - ref_off
        elif elem.name == "ChicaneSlit" and hasattr(elem, "misalignment"):
            # the slit camera is chamber-mounted too: shift the Screen so
            # its image reads in lab-frame coordinates (beam walks across
            # the camera as R56 changes, as on the machine)
            ref_off = _slit_ref_offset(sim, st)
            _assign(elem, "misalignment", [-ref_off, 0.0])
        elif hasattr(elem, "bfield_t"):  # the magspec
            elem.enabled = bool(st["magspec_on"])
            elem.bfield_t = float(st["magspec_b"])
        elif "chicane_sign" in md:
            # The coil current defines the FIELD (measured B-I table); the
            # actual bend angle follows from the beam rigidity at the
            # element. R56 is a derived readback, not a knob.
            field = cal.chicane_current_to_field(st["chicane_current"])
            angle = (
                md["chicane_sign"]
                * field
                * cal.CHICANE_BEND_LENGTH_M
                / cal.rigidity(energy)
            )
            _assign(elem, "angle", angle)


class MachineStateVariable(TorchScalarVariable, WritableActionMixin):
    """A machine-unit knob (coil amps, R56 um) backed by machine_state."""

    key: str

    def _get(self, sim):
        return torch.tensor(sim.machine_state[self.key])

    def _set(self, sim, value):
        sim.machine_state[self.key] = float(value)
        _apply_machine_state(sim)


class MachineStateBoolVariable(BoolVariable, WritableActionMixin):
    """A machine on/off toggle backed by machine_state (stored as 0/1)."""

    key: str

    def _get(self, sim):
        return bool(sim.machine_state[self.key])

    def _set(self, sim, value):
        sim.machine_state[self.key] = 1.0 if value else 0.0
        _apply_machine_state(sim)


def _visa_bump_geometry(sim) -> tuple[float, float]:
    """Lever arms (L1: S3->S4, L2: S4->VISA entrance) from the built lattice.

    Valid because S3 -> VISA entrance is drifts only (magspec off; when the
    magspec is on the downstream beam is dumped anyway).
    """
    if not hasattr(sim, "_bump_geom"):
        s = 0.0
        pos = {}
        for e in sim.segment.elements:
            pos[e.name] = s
            s += float(e.length)
        target = pos["DriftToUndulator"] + float(
            getattr(sim.segment, "DriftToUndulator").length
        )
        sim._bump_geom = (pos["S4H"] - pos["S3H"], target - pos["S4H"])
    return sim._bump_geom


class VisaBumpVariable(TorchScalarVariable, WritableActionMixin):
    """Composite two-corrector bump at the VISA entrance — a writable VIEW.

    Not new state: an exact, invertible coordinate transform on the raw
    (S3, S4) corrector currents in machine_state. Readbacks always derive
    from the currents, so hand-tuning S3/S4 shows up here honestly.
    Delta_x = th3*(L1+L2) + th4*L2 ; Delta_x' = th3 + th4.
    Excluded from config snapshots (raw currents are authoritative).
    Requested values are clamped to what the +/-5 A supplies can deliver —
    the readback reports the ACHIEVED value, like a real control system.
    """

    axis: str  # "H" or "V"
    mode: str  # "pos" (mm) or "angle" (mrad)

    def _angles(self, sim):
        st = sim.machine_state
        e = float(sim.energies["S3H"])
        th3 = cal.kicker_current_to_angle(
            st[f"S3{self.axis}"], cal.S_KICKER_MAX_CURRENT_A,
            cal.S_KICKER_MAX_INT_FIELD_TM, e)
        th4 = cal.kicker_current_to_angle(
            st[f"S4{self.axis}"], cal.S_KICKER_MAX_CURRENT_A,
            cal.S_KICKER_MAX_INT_FIELD_TM, e)
        return th3, th4, e

    def _get(self, sim):
        L1, L2 = _visa_bump_geometry(sim)
        th3, th4, _ = self._angles(sim)
        if self.mode == "pos":
            return torch.tensor((th3 * (L1 + L2) + th4 * L2) * 1e3)
        return torch.tensor((th3 + th4) * 1e3)

    def _set(self, sim, value):
        L1, L2 = _visa_bump_geometry(sim)
        th3, th4, e = self._angles(sim)
        x = th3 * (L1 + L2) + th4 * L2
        xp = th3 + th4
        if self.mode == "pos":
            x = float(value) * 1e-3
        else:
            xp = float(value) * 1e-3
        th3 = (x - xp * L2) / L1
        th4 = xp - th3
        imax = cal.S_KICKER_MAX_CURRENT_A
        for key, th in ((f"S3{self.axis}", th3), (f"S4{self.axis}", th4)):
            amp = cal.kicker_angle_to_current(
                th, imax, cal.S_KICKER_MAX_INT_FIELD_TM, e)
            sim.machine_state[key] = max(-imax, min(imax, amp))
        _apply_machine_state(sim)


class ChicaneR56Readback(TorchScalarVariable, ReadOnlyActionMixin):
    """Derived chicane R56 [um] (100 MeV ops convention) from the coil
    current via the measured B-I table. Readback only — the knob is
    ChicaneDipole_Current."""

    def _get(self, sim):
        field = cal.chicane_current_to_field(sim.machine_state["chicane_current"])
        return torch.tensor(cal.chicane_field_to_r56_um(field))


def _commit_source_param(sim, param: str, value) -> None:
    """Transactionally set one source parameter and regenerate the beam.

    Build the beam BEFORE committing anything to the simulator, so an
    invalid parameter fails this put without poisoning state (a poisoned
    state also re-crashes the Runner's reset-to-cached-state recovery
    path — learned the hard way).
    """
    from dataclasses import replace

    params = replace(sim.source_params)
    setattr(params, param, value)
    beam = make_beam(params)  # may raise; simulator untouched if so
    sim.source_params = params
    sim.initial_beam_distribution = beam.clone()
    sim.initial_beam_distribution_charge = beam.particle_charges.clone()
    sim.beam_distribution = beam.clone()
    sim.energies = sim.get_energy()
    _apply_machine_state(sim)  # magnets hold their currents/fields


class SourceVariable(TorchScalarVariable, WritableActionMixin):
    """A source-distribution parameter; puts regenerate the initial beam."""

    param: str

    def _get(self, sim):
        return torch.tensor(getattr(sim.source_params, self.param))

    def _set(self, sim, value):
        _commit_source_param(sim, self.param, float(value))


class SourceIntVariable(IntVariable, WritableActionMixin):
    """An integer source parameter (e.g. macro-particle count)."""

    param: str

    def _get(self, sim):
        return int(getattr(sim.source_params, self.param))

    def _set(self, sim, value):
        _commit_source_param(sim, self.param, int(value))


def build_writable_variables() -> list:
    """All read-write action variables (the machine + source knobs).

    Module-level so tooling (htu/config_tool.py snapshots) can enumerate
    the writable PV surface WITHOUT building the model — variable objects
    are cheap declarative pydantic models. Kept DRY: build_htu_model uses
    this exact list.
    """
    variables = [
        MachineStateVariable(
            name=f"{name}_Current", key=name, unit="A", value_range=(-10.0, 10.0)
        )
        for name in EMQ_DESIGNS
    ]
    variables.append(
        MachineStateVariable(
            name="ChicaneDipole_Current", key="chicane_current", unit="A",
            value_range=(0.0, 20.0),
        )
    )
    variables += [
        MachineStateVariable(
            name="ChicaneSlit_Jaw1_mm", key="chicane_jaw1_mm", unit="mm",
            value_range=(-20.0, 20.0),
        ),
        MachineStateVariable(
            name="ChicaneSlit_Jaw2_mm", key="chicane_jaw2_mm", unit="mm",
            value_range=(-20.0, 20.0),
        ),
    ]
    variables += [
        MachineStateVariable(
            name=f"{s}{p}_Current", key=f"{s}{p}", unit="A", value_range=(-5.0, 5.0)
        )
        for s in ["S1", "S2", "S3", "S4"]
        for p in ["H", "V"]
    ]
    variables += [
        MachineStateVariable(
            name=f"VS{i}{p}_Current", key=f"VS{i}{p}", unit="A",
            value_range=(-5.0, 5.0),
        )
        for i in range(1, 9)
        for p in ["H", "V"]
    ]
    variables += [
        MachineStateVariable(
            name="PMQTriplet_X_mm", key="pmq_x_mm", unit="mm", value_range=(-5.0, 5.0)
        ),
        MachineStateVariable(
            name="PMQTriplet_Y_mm", key="pmq_y_mm", unit="mm", value_range=(-5.0, 5.0)
        ),
        MachineStateVariable(
            name="PMQTriplet_Z_mm", key="pmq_z_mm", unit="mm", value_range=(-25.0, 25.0)
        ),
        MachineStateVariable(
            name="PMQ1V_Tilt_mrad", key="pmq1v_tilt_mrad", unit="mrad",
            value_range=(-200.0, 200.0),
        ),
        MachineStateVariable(
            name="PMQ2H_Tilt_mrad", key="pmq2h_tilt_mrad", unit="mrad",
            value_range=(-200.0, 200.0),
        ),
        MachineStateVariable(
            name="PMQ3V_Tilt_mrad", key="pmq3v_tilt_mrad", unit="mrad",
            value_range=(-200.0, 200.0),
        ),
        MachineStateBoolVariable(name="MagSpec_On", key="magspec_on"),
        MachineStateVariable(
            name="MagSpec_B_T", key="magspec_b", unit="T", value_range=(0.0, 1.65)
        ),
    ]
    variables += [
        SourceVariable(name="Source_Energy_MeV", param="energy_mev",
                       unit="MeV", value_range=(20.0, 1000.0)),
        SourceVariable(name="Source_EnergySpread_pct", param="energy_spread_pct",
                       unit="%", value_range=(0.0, 30.0)),
        SourceVariable(name="Source_Charge_pC", param="charge_pc",
                       unit="pC", value_range=(0.0, 1000.0)),
        SourceVariable(name="Source_BetaX_mm", param="beta_x_mm",
                       unit="mm", value_range=(0.0, 1000.0)),
        SourceVariable(name="Source_AlphaX", param="alpha_x",
                       unit="", value_range=(-20.0, 20.0)),
        SourceVariable(name="Source_NormEmitX_um", param="norm_emit_x_um",
                       unit="um", value_range=(0.0, 100.0)),
        SourceVariable(name="Source_BetaY_mm", param="beta_y_mm",
                       unit="mm", value_range=(0.0, 1000.0)),
        SourceVariable(name="Source_AlphaY", param="alpha_y",
                       unit="", value_range=(-20.0, 20.0)),
        SourceVariable(name="Source_NormEmitY_um", param="norm_emit_y_um",
                       unit="um", value_range=(0.0, 100.0)),
        SourceVariable(name="Source_X_um", param="x_um",
                       unit="um", value_range=(-500.0, 500.0)),
        SourceVariable(name="Source_Y_um", param="y_um",
                       unit="um", value_range=(-500.0, 500.0)),
        SourceVariable(name="Source_Xp_mrad", param="xp_mrad",
                       unit="mrad", value_range=(-10.0, 10.0)),
        SourceVariable(name="Source_Yp_mrad", param="yp_mrad",
                       unit="mrad", value_range=(-10.0, 10.0)),
        # macro-particle count: statistics vs re-track speed (~0.2s at 20k)
        SourceIntVariable(name="Source_NumParticles", param="num_particles",
                          value_range=(100, 500_000)),
    ]
    return variables


def writable_pv_names() -> list[str]:
    """PV suffixes (= variable names) of every read-write PV."""
    return [v.name for v in build_writable_variables()]


def boolean_pv_names() -> set[str]:
    """The writable PVs whose values are booleans (save/load as true/false)."""
    return {
        v.name for v in build_writable_variables() if isinstance(v, BoolVariable)
    }


def integer_pv_names() -> set[str]:
    """The writable PVs whose values are integers (e.g. macro-particle count)."""
    return {
        v.name for v in build_writable_variables() if isinstance(v, IntVariable)
    }


def build_htu_model(
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
    to_element: str | None = None,  # full line incl. undulator FODO
    source_params: SourceParams | None = None,
) -> LUMECheetahModel:
    segment = build_htu_segment(
        screen_geometries=screen_geometries,
        active_screens=active_screens,
        to_element=to_element,
    )
    params = source_params or SourceParams()
    simulator = CheetahSimulator(
        segment=segment, initial_beam_distribution=make_beam(params)
    )
    simulator.source_params = params
    simulator.machine_state = dict(DEFAULT_MACHINE_STATE)
    _apply_machine_state(simulator)
    simulator.track()  # re-render screens with applied state (e.g. the
    # slit camera's lab-frame misalignment) before the model reads them

    variables = build_writable_variables()

    from htu.magspec import FRONT_SHAPE, SIDE_SHAPE, SIDE_ZOOM_SHAPE

    variables += [
        CheetahReadOnlyNDVariable(
            name="MagSpec_FrontScreen_image", element_name="MagSpec",
            element_attribute="front_reading", unit="counts", shape=FRONT_SHAPE,
        ),
        CheetahReadOnlyNDVariable(
            name="MagSpec_SideScreen_image", element_name="MagSpec",
            element_attribute="side_reading", unit="counts", shape=SIDE_SHAPE,
        ),
        # auto-tracking zoom: +/-50mm window recentered on the beam centroid
        CheetahReadOnlyNDVariable(
            name="MagSpec_SideZoom_image", element_name="MagSpec",
            element_attribute="side_zoom_reading", unit="counts",
            shape=SIDE_ZOOM_SHAPE,
        ),
        CheetahReadOnlyScalarVariable(
            name="MagSpec_SideZoom_Center_mm", element_name="MagSpec",
            element_attribute="side_zoom_center_mm", unit="mm", read_only=True,
        ),
    ChicaneR56Readback(name="Chicane_R56_um", unit="um", read_only=True),
    ]

    # composite bump knobs: writable VIEWS over (S3, S4) currents —
    # deliberately NOT in build_writable_variables, so config snapshots
    # carry only the authoritative raw currents
    variables += [
        VisaBumpVariable(name="VisaBump_X_mm", axis="H", mode="pos",
                         unit="mm", value_range=(-4.0, 4.0)),
        VisaBumpVariable(name="VisaBump_Xp_mrad", axis="H", mode="angle",
                         unit="mrad", value_range=(-2.0, 2.0)),
        VisaBumpVariable(name="VisaBump_Y_mm", axis="V", mode="pos",
                         unit="mm", value_range=(-4.0, 4.0)),
        VisaBumpVariable(name="VisaBump_Yp_mrad", axis="V", mode="angle",
                         unit="mrad", value_range=(-2.0, 2.0)),
    ]

    geo = screen_geometries or {}
    for name in active_screens:
        g = geo.get(name, DEFAULT_GEOMETRY)
        variables.append(
            CheetahReadOnlyNDVariable(
                name=f"{name}_image",
                element_name=name,
                element_attribute="reading",
                unit="counts",
                shape=(g.size_y, g.size_x),
            )
        )

    return LUMECheetahModel(simulator=simulator, action_variables=variables)
