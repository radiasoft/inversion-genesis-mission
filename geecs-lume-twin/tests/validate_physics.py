"""Physics validation for the HTU twin — the checks this project was
accepted against, consolidated. Run headless (no server needed):

    .venv/bin/python tests/validate_physics.py

Exits nonzero on any failure. Each check states its physics rationale.
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

from htu.magspec import SIDE_RANGE_M, SIDE_SHAPE, DutchCalibration  # noqa: E402
from htu.model import build_htu_model  # noqa: E402
from htu import calibrations as hcal  # noqa: E402
from htu.source import SourceParams, make_beam  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not cond:
        FAILURES.append(name)


def img_stats(img, pix_mm=0.007, center=512):
    img = np.asarray(img)
    tot = img.sum()
    ys, xs = np.indices(img.shape)
    cx = (img * xs).sum() / tot
    sx = np.sqrt((img * (xs - cx) ** 2).sum() / tot) * pix_mm
    ysig = np.sqrt(
        (img * (ys - (img * ys).sum() / tot) ** 2).sum() / tot
    ) * pix_mm
    return (cx - center) * pix_mm, sx, ysig


def side_spectrum(img):
    xs = np.linspace(*SIDE_RANGE_M, SIDE_SHAPE[1]) * 1000
    proj = np.asarray(img).sum(axis=0)
    cen = (proj * xs).sum() / proj.sum()
    rms = np.sqrt((proj * (xs - cen) ** 2).sum() / proj.sum())
    return cen, rms


# --- 1. Calibration-level: pencils land at the table positions ------------
cal = DutchCalibration()
for p, screen in [(50.0, "side"), (121.2, "side"), (400.0, "front"), (800.0, "front")]:
    n = 200
    r = cal.map_to_screens(
        np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, p), np.ones(n)
    )
    s, y, q = r[screen]
    tbl = np.interp(
        p,
        cal.p_mev[: cal.n_rows],
        (cal.front_screen_m if screen == "front" else cal.side_screen_m)[: cal.n_rows],
    )
    check(
        f"magspec pencil p_eff={p} lands at table position ({screen})",
        s.size == n and abs(np.mean(s) - tbl) < 1e-6,
        f"landed {np.mean(s)*1000:.2f} mm vs table {tbl*1000:.2f} mm",
    )
check("front/side split near doc value (~290-330 MeV/c)", 250 < cal.p_side_max < 330,
      f"split at {cal.p_side_max:.0f} MeV/c")

# --- 2. Model-level ------------------------------------------------------
m = build_htu_model()
Q0 = 30e-12


def aline_q():
    return float(np.asarray(m._state["UC_ALineEbeam1_image"]).sum())


check("EMQ current readbacks invert calibration exactly",
      abs(float(m._state["EMQ1H_Current"]) - 0.683822195172598) < 1e-6)
check("full transmission at nominal settings", abs(aline_q() / Q0 - 1) < 0.005,
      f"{aline_q()/Q0*100:.1f}%")

# chromatic blur: shrinking energy spread shrinks the Aline y-size
m._set({"Source_EnergySpread_pct": 0.2})
_, _, y_cold = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"Source_EnergySpread_pct": 5.0})
_, _, y_hot = img_stats(m._state["UC_ALineEbeam1_image"])
check("chromatic blur present (5% vs 0.2% spread)", y_hot > y_cold * 1.05,
      f"y_rms {y_hot:.3f} vs {y_cold:.3f} mm")

# energy jitter: currents hold, optics shift
m._set({"Source_Energy_MeV": 110.0})
rb = float(m._state["EMQ1H_Current"])
_, sx110, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"Source_Energy_MeV": 100.0})
_, sx100, _ = img_stats(m._state["UC_ALineEbeam1_image"])
check("magnet currents hold through source-energy change",
      abs(rb - 0.683822195172598) < 1e-6)
check("optics shift with source energy at fixed currents",
      abs(sx110 - sx100) / sx100 > 0.05, f"x_rms {sx100:.3f} -> {sx110:.3f} mm")

# steering
m._set({"S1H_Current": 2.0})
cx, _, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"S1H_Current": 0.0})
check("S1H 2A moves Aline1 centroid ~+3.2mm", 2.5 < cx < 4.0, f"{cx:+.2f} mm")

# --- 3. Magspec through the model ---------------------------------------
m._set({"MagSpec_On": True, "Source_EnergySpread_pct": 0.1})
cen8, _ = side_spectrum(m._state["MagSpec_SideScreen_image"])
q_ms = float(np.asarray(m._state["MagSpec_SideScreen_image"]).sum())
check("magspec on: full charge on side screen, downstream dark",
      abs(q_ms / Q0 - 1) < 0.01 and aline_q() < Q0 * 0.01,
      f"side={q_ms/Q0*100:.1f}%, Aline={aline_q()/Q0*100:.2f}%")
m._set({"MagSpec_B_T": 0.4})
cen4, _ = side_spectrum(m._state["MagSpec_SideScreen_image"])
check("spectral line moves strongly with B", abs(cen4 - cen8) > 50,
      f"{cen8:.0f} -> {cen4:.0f} mm")
m._set({"MagSpec_B_T": 0.825})

# dispersion linearity: spectrum rms scales with energy spread
m._set({"Source_EnergySpread_pct": 5.0})
_, rms5 = side_spectrum(m._state["MagSpec_SideScreen_image"])
m._set({"Source_EnergySpread_pct": 1.0})
_, rms1 = side_spectrum(m._state["MagSpec_SideScreen_image"])
check("spectrum rms scales ~linearly with energy spread (5:1)",
      3.5 < rms5 / rms1 < 6.5, f"ratio {rms5/rms1:.2f}")

# steering into the magspec = apparent momentum shift, no charge loss
m._set({"Source_EnergySpread_pct": 0.1, "S3H_Current": 3.0})
cen_steer, _ = side_spectrum(m._state["MagSpec_SideScreen_image"])
q_steer = float(np.asarray(m._state["MagSpec_SideScreen_image"]).sum())
check("S3 steering shifts apparent momentum without charge loss",
      abs(q_steer / Q0 - 1) < 0.01 and 1.0 < abs(cen_steer - cen8) < 10.0,
      f"shift {cen_steer - cen8:+.1f} mm, charge {q_steer/Q0*100:.1f}%")
m._set({"S3H_Current": 0.0, "MagSpec_On": False, "Source_EnergySpread_pct": 5.0})

# --- 4. Apertures --------------------------------------------------------
m._set({"S2H_Current": 3.0})
check("steering into the EMQ pipe wall loses the beam", aline_q() < Q0 * 0.05,
      f"transmission {aline_q()/Q0*100:.1f}%")
m._set({"S2H_Current": 0.0})
m._set({"EMQ2V_Current": 0.0})
t = aline_q() / Q0
m._set({"EMQ2V_Current": -0.882403221217054})
check("mismatched triplet scrapes partially", 0.7 < t < 0.99, f"{t*100:.1f}%")
check("state fully restored after all checks", abs(aline_q() / Q0 - 1) < 0.005)

# --- 5. PMQ triplet hexapod ----------------------------------------------
m._set({"PMQTriplet_X_mm": 0.05})
cx, _, _ = img_stats(m._state["UC_ALineEbeam1_image"])
check("hexapod X 50um steers Aline1 (~11 mm/mm sensitivity)",
      0.3 < cx < 0.9, f"{cx:+.2f} mm")
m._set({"PMQTriplet_X_mm": 0.5})
check("hexapod X 0.5mm scrapes the beam on the pipe", aline_q() < Q0 * 0.1,
      f"transmission {aline_q()/Q0*100:.0f}%")
m._set({"PMQTriplet_X_mm": 0.0, "PMQTriplet_Z_mm": -10.0})
_, sxm, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"PMQTriplet_Z_mm": 10.0})
_, sxp, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"PMQTriplet_Z_mm": 0.0})
_, sx0, _ = img_stats(m._state["UC_ALineEbeam1_image"])
check("hexapod Z walks the focus (spot size monotonic over -10..+10mm)",
      sxm < sx0 < sxp, f"sx {sxm:.3f} < {sx0:.3f} < {sxp:.3f} mm")
check("hexapod restored to baseline", abs(aline_q() / Q0 - 1) < 0.005)

# --- 6. Twiss source parameterization ------------------------------------
# defaults reproduce the old (spot 5um, div 1mrad) beam to ~1%: beta 5mm,
# norm emit 1.0 um at 100 MeV -> eps_geom 5.11e-9 vs the old 5e-9
b = make_beam(SourceParams())
sx_src = float(b.particles[..., 0].std()) * 1e6
check("default Twiss source reproduces old beam (~5.06 um rms spot)",
      4.9 < sx_src < 5.2, f"sigma_x {sx_src:.3f} um")

# pointing: Xp=0.5 mrad moves the Aline1 centroid measurably and restores
# (sign flips through the quad optics — magnitude is the physics claim)
m._set({"Source_Xp_mrad": 0.5})
cx_pt, _, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"Source_Xp_mrad": 0.0})
cx_pt0, _, _ = img_stats(m._state["UC_ALineEbeam1_image"])
check("source pointing 0.5 mrad moves Aline1 centroid (~0.1 mm) + restores",
      0.05 < abs(cx_pt - cx_pt0) < 0.3 and abs(cx_pt0) < 0.05,
      f"shift {cx_pt - cx_pt0:+.3f} mm, restored {cx_pt0:+.4f} mm")

# emittance: doubling norm emit x grows the Aline1 x spot ~sqrt(2)
_, sx_e1, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"Source_NormEmitX_um": 2.0})
_, sx_e2, _ = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"Source_NormEmitX_um": 1.0})
check("doubling norm emit x grows Aline1 x spot (~sqrt2)",
      1.2 < sx_e2 / sx_e1 < 1.6, f"ratio {sx_e2/sx_e1:.2f}")

# adiabatic damping: at fixed NORMALIZED emittance the geometric source
# spot shrinks as 1/sqrt(gamma). Checked at the source: downstream the
# direction reverses (fixed magnet currents -> mismatch dominates,
# verified empirically), so Aline1 is not a clean damping observable.
sx_100 = float(make_beam(SourceParams(energy_mev=100.0)).particles[..., 0].std())
sx_400 = float(make_beam(SourceParams(energy_mev=400.0)).particles[..., 0].std())
check("adiabatic damping: 4x energy halves geometric source spot",
      0.45 < sx_400 / sx_100 < 0.55, f"ratio {sx_400/sx_100:.3f}")

# --- 7. PMQ tilt (skew coupling) ------------------------------------------
m._set({"PMQ2H_Tilt_mrad": 100.0})
_, sx_t, sy_t = img_stats(m._state["UC_ALineEbeam1_image"])
m._set({"PMQ2H_Tilt_mrad": 0.0})
_, sx_t0, sy_t0 = img_stats(m._state["UC_ALineEbeam1_image"])
check("PMQ2H tilt 100 mrad couples the planes (both spots grow) + restores",
      sx_t > sx_t0 * 1.5 and sy_t > sy_t0 * 1.3
      and abs(aline_q() / Q0 - 1) < 0.005,
      f"sx {sx_t0:.3f}->{sx_t:.3f}, sy {sy_t0:.3f}->{sy_t:.3f} mm")

# --- 8. Chicane energy slit (LAB-frame jaw coordinates) -------------------
# Jaw PVs are lab-frame positions relative to the STRAIGHT (R56=0) axis —
# the physical plates are chamber-mounted. The reference orbit at the slit
# sits ref_offset off that axis (~6.1 mm at R56=200um), so the beam walks
# across a fixed slit as R56 changes. Dispersion ~6.3 mm per unit dp/p,
# LOW energy at +x: jaw1 (+x) cuts the low-energy tail.
from htu.model import _slit_ref_offset  # noqa: E402

check("slit jaws retracted = baseline transmission",
      abs(aline_q() / Q0 - 1) < 0.005, f"{aline_q()/Q0*100:.1f}%")

beam_lab = _slit_ref_offset(m.simulator, m.simulator.machine_state) * 1000
check("reference orbit offset at slit ~6.1mm at R56=200um",
      5.5 < beam_lab < 6.7, f"{beam_lab:.2f} mm")

r56_rb = float(m._state["Chicane_R56_um"])
check("R56 readback derives from coil current via measured B-I table",
      abs(r56_rb - 200.0) < 2.0, f"{r56_rb:.1f} um at 2.1688 A")


def slit_cam_cx(pix_mm=0.020):
    img = np.asarray(m._state["ChicaneSlit_image"])
    xs = np.indices(img.shape)[1]
    return ((img * xs).sum() / img.sum() - 512) * pix_mm


check("slit camera reads lab frame (centroid = ref offset at R56=200)",
      abs(slit_cam_cx() - beam_lab) < 0.5, f"cam {slit_cam_cx():+.2f} mm")
m._set({"ChicaneDipole_Current": hcal.chicane_r56_to_current_a(50.0)})
check("beam walks across the slit camera with R56 (200 -> 50um)",
      2.5 < slit_cam_cx() < 3.6, f"cam {slit_cam_cx():+.2f} mm")
m._set({"ChicaneDipole_Current": 2.1688})

# a fixed slit on the straight axis: blocks at R56=200, passes at R56=5
m._set({"ChicaneSlit_Jaw1_mm": 1.5, "ChicaneSlit_Jaw2_mm": -1.5})
t_high = aline_q() / Q0
m._set({"ChicaneDipole_Current": hcal.chicane_r56_to_current_a(5.0)})
t_low = aline_q() / Q0
m._set({"ChicaneDipole_Current": 2.1688})
check("fixed on-axis slit: beam blocked at R56=200, passes at R56=5",
      t_high < 0.02 and t_low > 0.95,
      f"T(200)={t_high*100:.1f}%, T(5)={t_low*100:.1f}%")

# slit centered on the DISPLACED beam cuts the energy tails
m._set({"ChicaneSlit_Jaw1_mm": beam_lab + 0.5, "ChicaneSlit_Jaw2_mm": beam_lab - 0.5})
t_slit = aline_q() / Q0
check("+/-0.5mm slit centered on the beam cuts transmission",
      0.5 < t_slit < 0.95, f"{t_slit*100:.1f}%")
m._set({"ChicaneSlit_Jaw1_mm": 20.0, "ChicaneSlit_Jaw2_mm": -20.0})

m._set({"MagSpec_On": True})  # 5% spread still set from baseline
cen_open, rms_open = side_spectrum(m._state["MagSpec_SideScreen_image"])
m._set({"ChicaneSlit_Jaw1_mm": beam_lab + 0.5, "ChicaneSlit_Jaw2_mm": beam_lab - 0.5})
cen_slit, rms_slit = side_spectrum(m._state["MagSpec_SideScreen_image"])
check("closing the slit narrows the magspec spectrum (energy selection)",
      rms_slit < rms_open * 0.9,
      f"rms {rms_open:.1f} -> {rms_slit:.1f} mm")
m._set({"ChicaneSlit_Jaw1_mm": beam_lab, "ChicaneSlit_Jaw2_mm": -20.0})
cen_j1, _ = side_spectrum(m._state["MagSpec_SideScreen_image"])
check("single +x jaw at beam center cuts the LOW-energy side",
      cen_j1 > cen_open + 3.0, f"cen {cen_open:.1f} -> {cen_j1:.1f} mm")
try:
    m._set({"ChicaneSlit_Jaw1_mm": beam_lab - 0.5,
            "ChicaneSlit_Jaw2_mm": beam_lab + 0.5})
    q_closed = float(np.asarray(m._state["MagSpec_SideScreen_image"]).sum())
    ok = q_closed == 0.0
except Exception:
    ok = False
check("crossed jaws = fully closed, zero charge, no crash", ok)
m._set({"ChicaneSlit_Jaw1_mm": 20.0, "ChicaneSlit_Jaw2_mm": -20.0,
        "MagSpec_On": False})
check("slit reopened restores baseline", abs(aline_q() / Q0 - 1) < 0.005)

# --- 8b. Undulator FODO line ----------------------------------------------
visa8_q = float(np.asarray(m._state["UC_VisaEBeam8_image"]).sum())
check("full transmission through the four VISA FODO segments",
      visa8_q / Q0 > 0.95, f"{visa8_q/Q0*100:.1f}% at VisaEBeam8")
# VS1 sits immediately upstream of VisaEBeam1 (zero drift): a kick there
# changes angle, not position, at that screen — measure one screen down.
m._set({"VS1H_Current": 3.0})
img = np.asarray(m._state["UC_VisaEBeam2_image"])
cx_vs = ((img * np.indices(img.shape)[1]).sum() / img.sum() - 512) * 0.007
m._set({"VS1H_Current": 0.0})
check("VS1H steering moves the downstream VisaEBeam2 centroid",
      abs(cx_vs) > 0.2, f"{cx_vs:+.2f} mm at 3A")

# --- 8c. VISA entrance bump composites ------------------------------------
m._set({"VisaBump_X_mm": 0.5})
st = m.simulator.machine_state
check("position bump decomposes to equal-opposite S3/S4 currents",
      abs(st["S3H"] - 0.5) < 0.01 and abs(st["S4H"] + 0.5) < 0.01,
      f"S3H={st['S3H']:+.3f} S4H={st['S4H']:+.3f} A")
check("position bump leaves entrance angle at zero",
      abs(float(m._state["VisaBump_Xp_mrad"])) < 1e-3,
      f"Xp={float(m._state['VisaBump_Xp_mrad']):+.4f} mrad")
m._set({"VisaBump_X_mm": 0.0, "VisaBump_Xp_mrad": 0.3})
check("angle bump leaves entrance position at zero",
      abs(float(m._state["VisaBump_X_mm"])) < 1e-3,
      f"X={float(m._state['VisaBump_X_mm']):+.4f} mm")
m._set({"VisaBump_Xp_mrad": 0.0})
check("bump knobs restore raw currents to zero",
      abs(st["S3H"]) < 1e-6 and abs(st["S4H"]) < 1e-6)

# --- 9. Macro-particle count ---------------------------------------------
m._set({"Source_NumParticles": 5000})
check("changing macro-particle count conserves total charge",
      abs(aline_q() / Q0 - 1) < 0.02, f"{aline_q()/Q0*100:.1f}% at N=5000")
m._set({"Source_NumParticles": 20000})
check("particle count restored", int(m._state["Source_NumParticles"]) == 20000)

# --- 10. Crash regression ------------------------------------------------
try:
    m._set({"Source_EnergySpread_pct": 0.0})
    ok = True
    m._set({"Source_EnergySpread_pct": 5.0})
except Exception:
    ok = False
check("0% energy spread does not crash (cold beam)", ok)

try:
    m._set({"Source_NormEmitX_um": 0.0, "Source_NormEmitY_um": 0.0})
    m._set({"Source_BetaX_mm": 0.0, "Source_BetaY_mm": 0.0})
    ok = abs(aline_q() / Q0 - 1) < 0.01  # point-like beam still transports
    m._set({"Source_NormEmitX_um": 1.0, "Source_NormEmitY_um": 1.0,
            "Source_BetaX_mm": 5.0, "Source_BetaY_mm": 5.0})
except Exception:
    ok = False
check("zero emittance/beta does not crash (point-like beam)", ok)

print(f"\n{len(FAILURES)} failures" if FAILURES else "\nALL CHECKS PASSED")
sys.exit(1 if FAILURES else 0)
