"""Camera-accurate screen geometry for the HTU twin.

The experimental image size/resolution/registration trick is ported from
BLAST-AI-ML/bella_htu_digital_twin: parse the GEECS ECS Live Dump for each
camera device (SpatialCalibration, ROI, "Big Blue" crosshair) and misalign
the simulated Screen so the crosshair pixel sits on the beamline reference
axis — the synthetic image then overlays the real camera image pixel-for-
pixel.
"""

import re
from dataclasses import dataclass

import torch
from cheetah.accelerator import Screen


@dataclass
class ScreenGeometry:
    """Per-camera geometry in experimental units."""

    size_x: int  # pixels (image width)
    size_y: int  # pixels (image height)
    calibration_m: float  # meters per pixel
    crosshair_x: float | None = None  # pixels from image left
    crosshair_y: float | None = None  # pixels from image top

    @property
    def misalignment_m(self) -> tuple[float, float]:
        """Screen offset so the crosshair pixel lands on the reference axis.

        Sign conventions ported verbatim: crosshair_x measured from the left
        (axis at left edge -> positive misalignment_x), crosshair_y from the
        top (axis at top edge -> negative misalignment_y).
        """
        cx = self.crosshair_x if self.crosshair_x is not None else self.size_x / 2
        cy = self.crosshair_y if self.crosshair_y is not None else self.size_y / 2
        mis_x = -self.calibration_m * (cx - self.size_x / 2)
        mis_y = self.calibration_m * (cy - self.size_y / 2)
        return mis_x, mis_y


# Fallback used by the source repo when no ECS dump is available.
DEFAULT_GEOMETRY = ScreenGeometry(size_x=1024, size_y=1024, calibration_m=7e-6)


def make_screen(
    name: str,
    geometry: ScreenGeometry | None = None,
    is_active: bool = False,
    method: str = "histogram",
) -> Screen:
    """Build a Cheetah Screen with experimental camera geometry.

    method='histogram' keeps interactive serving fast; the source repo used
    'kde' (bandwidth 2x calibration) for smoother fitted images.
    """
    g = geometry or DEFAULT_GEOMETRY
    kwargs = {}
    if method == "kde":
        kwargs["kde_bandwidth"] = torch.tensor(2 * g.calibration_m)
    return Screen(
        name=name,
        resolution=(g.size_x, g.size_y),
        pixel_size=torch.tensor([g.calibration_m, g.calibration_m]),
        misalignment=torch.tensor(g.misalignment_m),
        binning=1,
        is_active=is_active,
        method=method,
        **kwargs,
    )


# --- ECS Live Dump parsing (ported from read_experiment_configuration.py) ---


def _device_block(device_name: str, text: str) -> str:
    pattern = rf'Device Name = "{device_name}"\n(.*?)(?:\n\s*\n|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected 1 ECS block for {device_name}, got {len(matches)}")
    return matches[0]


def _read_variable(block: str, variable_name: str) -> str:
    return re.findall(rf'{variable_name} = "(.*)"', block)[0]


def geometry_from_ecs_dump(
    device_name: str,
    ecs_dump_text: str,
    analysis_settings: dict | None = None,
) -> ScreenGeometry:
    """Extract a camera's geometry from a GEECS ECS Live Dump.

    analysis_settings, when given, is the per-device crop dict of the source
    repo ({'Left ROI', 'Top ROI', 'Size_X', 'Size_Y'}): the crosshair is
    re-referenced to the cropped image.
    """
    block = _device_block(device_name, ecs_dump_text)

    label1 = _read_variable(block, "Crosshair.Label1").lower()
    label2 = _read_variable(block, "Crosshair.Label2").lower()
    if "big blu" in label1:
        cx = float(_read_variable(block, "Target.X"))
        cy = float(_read_variable(block, "Target.Y"))
    elif "big blu" in label2:
        cx = float(_read_variable(block, "Target2.X"))
        cy = float(_read_variable(block, "Target2.Y"))
    else:
        raise ValueError(f"No Big Blue crosshair reference for {device_name}")

    calibration_um = float(_read_variable(block, "SpatialCalibration"))
    size_x = int(_read_variable(block, "ROI.Width"))
    size_y = int(_read_variable(block, "ROI.Height"))

    if analysis_settings is not None:
        s = analysis_settings[device_name]
        cx -= s["Left ROI"]
        cy -= s["Top ROI"]
        size_x, size_y = s["Size_X"], s["Size_Y"]

    return ScreenGeometry(
        size_x=size_x,
        size_y=size_y,
        calibration_m=calibration_um * 1e-6,
        crosshair_x=cx,
        crosshair_y=cy,
    )
