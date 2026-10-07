"""Dutch-magnet magspec as a map element, per docs/magspec-design.md.

Replays the 2013 Nakamura characterization: the trajectory table gives each
momentum's landing point on the front/side screens (computed at field-map
max = 1 T; momentum scales linearly with B), and the per-momentum 3rd-order
COSY maps give the beam-dynamics around each reference trajectory. Per
particle we interpolate across the map FAMILY at its own momentum (delta=0)
— right for full-spectrum devices; assumes the 1597-row family is dense
enough to carry the chromatic dependence (it is, by construction).

Off (enabled=False or B<=0): the element is a drift (transport mode).
On: deposits charge-weighted images on the two screens and zeroes the
outgoing charge, so the downstream line goes dark (characterization mode).
"""

import logging
import re
from pathlib import Path

import numpy as np
import torch
from cheetah.accelerator import Drift

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "dutch"
TRJ_FILE = DATA_DIR / "180702Dutch1Trj.txt"
MAP_DIR = DATA_DIR / "map"

# NOTE: the doc's ~+/-2.7 mrad "angular acceptance" is an upstream PIPE
# limit, not a property of the magnet — enforced by Aperture elements in
# the lattice (e.g. MagSpecEntrance_AP), NOT by an angle cut here. A flat
# angle cut wrongly eats deliberately steered beams (found via S3 scans).

# screen image geometry (screen-coordinate space, meters; conventions:
# front screen 0 = beam axis; side screen 0 = magnet center).
# PHYSICAL extents with SQUARE pixels, so images render in true aspect:
# - dispersed plane: front ~0-180 mm (Trj table + chamber), side +/-360 mm
#   (27.9" chamber, both per the Nakamura doc / 2018 setup slides)
# - non-dispersed (y): ~1" LANEX strip height (2018 camera views ~25-35 mm;
#   exact strip height is an assumption — adjust here if measured)
FRONT_PIX_M = 4.0e-4  # 0.4 mm/px, square
SIDE_PIX_M = 5.0e-4  # 0.5 mm/px, square
FRONT_SHAPE = (64, 450)  # (ny, nx): 25.6 x 180 mm
SIDE_SHAPE = (51, 1440)  # 25.5 x 720 mm
FRONT_RANGE_M = (0.0, FRONT_SHAPE[1] * FRONT_PIX_M)
SIDE_RANGE_M = (-SIDE_SHAPE[1] * SIDE_PIX_M / 2, SIDE_SHAPE[1] * SIDE_PIX_M / 2)
FRONT_Y_RANGE_M = (-FRONT_SHAPE[0] * FRONT_PIX_M / 2, FRONT_SHAPE[0] * FRONT_PIX_M / 2)
SIDE_Y_RANGE_M = (-SIDE_SHAPE[0] * SIDE_PIX_M / 2, SIDE_SHAPE[0] * SIDE_PIX_M / 2)

# auto-tracking zoom on the side screen: a +/-ZOOM_HALF window recentered
# on the charge centroid each track (window center published as a PV)
SIDE_ZOOM_HALF_M = 0.050
SIDE_ZOOM_SHAPE = (SIDE_SHAPE[0], int(2 * SIDE_ZOOM_HALF_M / SIDE_PIX_M))  # (51, 200)


def _parse_map_file(path: Path) -> dict[str, np.ndarray]:
    """One COSY map file -> {exponent_string: 5 output coefficients}."""
    terms = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6 or not re.fullmatch(r"\d{6}", parts[5]):
            continue  # terminator / blank
        terms[parts[5]] = np.array([float(v) for v in parts[:5]])
    return terms


class DutchCalibration:
    """Parsed trajectory table + COSY map family for one Dutch magspec."""

    def __init__(self, trj_file: Path = TRJ_FILE, map_dir: Path = MAP_DIR):
        rows = [
            line.split("\t")
            for line in trj_file.read_text().splitlines()
            if line.strip()
        ]
        header = [h.strip() for h in rows[0]]
        col = {name: i for i, name in enumerate(header)}
        data = np.array(
            [[float(v) for v in r[: len(header)]] for r in rows[1:]], dtype=float
        )

        self.p_mev = data[:, col["momentum [MeV/c]"]]
        self.side_logic = data[:, col["side logic"]]
        self.front_screen_m = data[:, col["front screen [m]"]]
        self.side_screen_m = data[:, col["side screen [m]"]]
        self.front_angle_deg = data[:, col["bending angle at front screen [dgr]"]]
        self.side_angle_deg = data[:, col["bending angle at side screen [dgr]"]]
        if not np.all(np.diff(self.p_mev) > 0):
            raise ValueError("trajectory table momenta are not ascending")

        map_files = sorted(map_dir.glob("*.txt"), key=lambda p: int(p.stem))
        n = min(len(map_files), len(self.p_mev))
        if len(map_files) != len(self.p_mev):
            logger.warning(
                "map/ has %d files but trajectory has %d rows; using first %d "
                "(alignment assumption: file %s <-> lowest-momentum row)",
                len(map_files), len(self.p_mev), n, map_files[0].name,
            )
        parsed = [_parse_map_file(f) for f in map_files[:n]]

        # dense coefficient arrays over the union of terms, restricted to
        # exponents in (x, a, y, b) only — we evaluate at t = delta = 0
        exps = sorted({e for t in parsed for e in t if e[4] == "0" and e[5] == "0"})
        self.exponents = np.array([[int(c) for c in e[:4]] for e in exps])  # (T,4)
        coeff = np.zeros((n, len(exps), 5))
        for i, t in enumerate(parsed):
            for j, e in enumerate(exps):
                if e in t:
                    coeff[i, j] = t[e]
        self.coeff_x = coeff[:, :, 0]  # x_f coefficients, (n, T)
        self.coeff_y = coeff[:, :, 2]  # y_f coefficients
        self.n_rows = n

        # side/front split: side logic is 1 for the low-momentum rows
        side_rows = np.nonzero(self.side_logic[:n] == 1.0)[0]
        self.p_side_max = float(self.p_mev[side_rows.max()]) if side_rows.size else 0.0

    def _interp_rows(self, arr: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Linear interpolation of per-row data (any trailing shape) at p."""
        grid = self.p_mev[: self.n_rows]
        idx = np.clip(np.searchsorted(grid, p) - 1, 0, self.n_rows - 2)
        w = (p - grid[idx]) / (grid[idx + 1] - grid[idx])
        w = np.clip(w, 0.0, 1.0)
        wshape = (-1,) + (1,) * (arr.ndim - 1)
        return arr[idx] * (1 - w.reshape(wshape)) + arr[idx + 1] * w.reshape(wshape)

    def map_to_screens(self, x, a, y, b, p_eff_mev, charge):
        """Apply the map family; return per-screen (s_m, y_m, charge) arrays.

        Inputs are numpy arrays at the reference plane; p_eff_mev is the
        1 T-equivalent momentum (p / B).
        """
        grid = self.p_mev[: self.n_rows]
        ok = (p_eff_mev >= grid[0]) & (p_eff_mev <= grid[-1])
        x, a, y, b, p, q = (v[ok] for v in (x, a, y, b, p_eff_mev, charge))
        if p.size == 0:
            return {"front": (np.empty(0),) * 3, "side": (np.empty(0),) * 3}

        # monomials (N, T) over (x, a, y, b)
        vars_ = np.stack([x, a, y, b], axis=1)  # (N, 4)
        mono = np.prod(
            vars_[:, None, :] ** self.exponents[None, :, :], axis=2
        )  # (N, T)
        cx = self._interp_rows(self.coeff_x, p)  # (N, T)
        cy = self._interp_rows(self.coeff_y, p)
        x_f = np.sum(cx * mono, axis=1)  # perpendicular to reference trajectory
        y_f = np.sum(cy * mono, axis=1)

        out = {}
        for screen, coord, angle in [
            ("front", self.front_screen_m, self.front_angle_deg),
            ("side", self.side_screen_m, self.side_angle_deg),
        ]:
            m = (p <= self.p_side_max) if screen == "side" else (p > self.p_side_max)
            s0 = self._interp_rows(coord, p[m])
            ang = np.radians(self._interp_rows(angle, p[m]))
            cos = np.cos(ang)  # angle is from the screen normal
            good = np.abs(cos) > 0.1  # drop grazing-incidence table edges
            out[screen] = (
                s0[good] + x_f[m][good] / cos[good],
                y_f[m][good],
                q[m][good],
            )
        return out


class DutchMagSpec(Drift):
    """The Dutch magspec: drift when off, screen-map element when on."""

    def __init__(self, name: str, length: float, calibration: DutchCalibration,
                 bfield_t: float = 0.825, enabled: bool = False):
        super().__init__(
            name=name,
            length=torch.tensor(length),
            tracking_method="drift_kick_drift",
        )
        self.calibration = calibration
        self.bfield_t = bfield_t
        self.enabled = enabled
        self.front_reading = torch.zeros(FRONT_SHAPE)
        self.side_reading = torch.zeros(SIDE_SHAPE)
        self.side_zoom_reading = torch.zeros(SIDE_ZOOM_SHAPE)
        self.side_zoom_center_mm = torch.tensor(0.0)

    @property
    def is_skippable(self) -> bool:
        # never let the Segment fold this into a transfer matrix
        return False

    def _histogram(self, s, y, q, s_range, y_range, shape):
        img, _, _ = np.histogram2d(
            y, s, bins=shape, range=[list(y_range), list(s_range)], weights=q
        )
        return torch.as_tensor(img, dtype=torch.float32)

    def track(self, incoming):
        if not self.enabled or self.bfield_t <= 0.0:
            self.front_reading = torch.zeros(FRONT_SHAPE)
            self.side_reading = torch.zeros(SIDE_SHAPE)
            self.side_zoom_reading = torch.zeros(SIDE_ZOOM_SHAPE)
            self.side_zoom_center_mm = torch.tensor(0.0)
            return super().track(incoming)

        e_ev = float(incoming.energy)
        m_ev = float(incoming.species.mass_eV)
        p0c_mev = np.sqrt(e_ev**2 - m_ev**2) / 1e6
        delta = incoming.p.detach().cpu().numpy().reshape(-1)
        p_eff = p0c_mev * (1.0 + delta) / self.bfield_t

        screens = self.calibration.map_to_screens(
            incoming.x.detach().cpu().numpy().reshape(-1),
            incoming.px.detach().cpu().numpy().reshape(-1),
            incoming.y.detach().cpu().numpy().reshape(-1),
            incoming.py.detach().cpu().numpy().reshape(-1),
            p_eff,
            # weight by survival: particles killed upstream (apertures,
            # the chicane slit jaws) must not appear in the spectrum
            (incoming.particle_charges * incoming.survival_probabilities)
            .detach().cpu().numpy().reshape(-1),
        )
        self.front_reading = self._histogram(
            *screens["front"], FRONT_RANGE_M, FRONT_Y_RANGE_M, FRONT_SHAPE
        )
        self.side_reading = self._histogram(
            *screens["side"], SIDE_RANGE_M, SIDE_Y_RANGE_M, SIDE_SHAPE
        )

        # auto-tracking zoom: +/-50 mm window recentered on the side-screen
        # charge centroid (window center published for the display readback)
        s_side, y_side, q_side = screens["side"]
        if q_side.sum() > 0:
            center = float(np.sum(s_side * q_side) / q_side.sum())
            self.side_zoom_reading = self._histogram(
                s_side, y_side, q_side,
                (center - SIDE_ZOOM_HALF_M, center + SIDE_ZOOM_HALF_M),
                SIDE_Y_RANGE_M, SIDE_ZOOM_SHAPE,
            )
            self.side_zoom_center_mm = torch.tensor(center * 1000.0)
        else:
            self.side_zoom_reading = torch.zeros(SIDE_ZOOM_SHAPE)
            self.side_zoom_center_mm = torch.tensor(0.0)

        # downstream goes dark: beam is bent out of the transport line
        outgoing = super().track(incoming)
        outgoing.particle_charges = torch.zeros_like(outgoing.particle_charges)
        return outgoing
