"""Generate htu_synoptic.bob from the actual lattice.

Walks build_htu_segment() accumulating s, and emits a to-scale cartoon:
clickable elements (EMQs -> quad controls, chicane bends -> R56, steering ->
H/V pair) plus a camera navigation-tab strip. Regenerate whenever the
lattice changes:  python display/generate_synoptic.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

from cheetah.accelerator import Dipole, HorizontalCorrector, Quadrupole, Screen, VerticalCorrector

from htu.lattice import build_htu_segment

# canvas layout
X0, WIDTH_PX = 30, 1440
AXIS_Y = 150
# (tab name, viewer file) — the magspec strips get true-aspect viewers
CAMERAS = [
    ("TCPhosphor", "camera_view.bob"),
    ("ChicaneSlit", "camera_view.bob"),
    ("DCPhosphor", "camera_view.bob"),
    ("Phosphor1", "camera_view.bob"),
    ("UC_ALineEbeam1", "camera_view.bob"),
    ("UC_ALineEBeam2", "camera_view.bob"),
    ("UC_ALineEBeam3", "camera_view.bob"),
    ("MagSpec_FrontScreen", "magspec_front_view.bob"),
    ("MagSpec_SideScreen", "magspec_side_view.bob"),
    *[(f"UC_VisaEBeam{i}", "camera_view.bob") for i in range(1, 9)],
]

COLORS = {
    "emq": (255, 140, 0),
    "pmq": (170, 170, 170),
    "bend": (60, 120, 255),
    "magspec": (120, 120, 200),
    "kicker": (200, 60, 200),
    "screen": (0, 190, 120),
}


def color_xml(rgb):
    r, g, b = rgb
    return f'<color red="{r}" green="{g}" blue="{b}" alpha="255"/>'


def label(text, x, y, w, h, size=10, rotated=False, bold=False):
    style = "BOLD" if bold else "REGULAR"
    rot = "<rotation_step>90.0</rotation_step>" if rotated else ""
    return f"""  <widget type="label" version="2.0.0">
    <name>lbl_{escape(text)}_{x}</name>
    <text>{escape(text)}</text>
    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>
    {rot}
    <font><font family="Liberation Sans" style="{style}" size="{size}.0"/></font>
  </widget>"""


def rect(x, y, w, h, rgb):
    return f"""  <widget type="rectangle" version="2.0.0">
    <name>rect_{x}_{y}</name>
    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>
    <background_color>{color_xml(rgb)}</background_color>
    <line_width>1</line_width>
  </widget>"""


def button(text, x, y, w, h, rgb, file, macros: dict, tooltip):
    macro_xml = "".join(f"<{k}>{escape(v)}</{k}>" for k, v in macros.items())
    return f"""  <widget type="action_button" version="3.0.0">
    <name>btn_{escape(text)}_{x}</name>
    <actions>
      <action type="open_display">
        <file>{file}</file>
        <macros>{macro_xml}</macros>
        <target>window</target>
        <description>{escape(tooltip)}</description>
      </action>
    </actions>
    <text>{escape(text)}</text>
    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>
    <background_color>{color_xml(rgb)}</background_color>
    <font><font family="Liberation Sans" style="REGULAR" size="8.0"/></font>
    <tooltip>{escape(tooltip)}</tooltip>
  </widget>"""


def main():
    seg = build_htu_segment()  # full line incl. undulator FODO
    total_length = sum(float(e.length) for e in seg.elements)
    px_per_m = WIDTH_PX / total_length

    widgets = []
    widgets.append(label("HTU transport twin — synoptic (click an element for controls)",
                         X0, 10, 900, 30, size=18, bold=True))
    # beam axis
    widgets.append(rect(X0, AXIS_Y - 1, WIDTH_PX, 2, (0, 0, 0)))

    # LPA source (upstream of everything): opens source-parameter controls
    widgets.append(button(
        "SRC", X0 - 28, AXIS_Y - 30, 26, 60, (255, 200, 0), "source_controls.bob",
        {}, "LPA source parameters (energy, spread, charge, spot, divergence)"))

    s = 0.0
    seen_kickers = set()
    kicker_widgets = []
    for elem in seg.elements:
        length = float(elem.length)
        x = X0 + int(s * px_per_m)
        w = max(int(length * px_per_m), 4)
        name = elem.name
        s += length

        if name == "MagSpec":
            widgets.append(button(
                "", x, AXIS_Y - 35, w, 70, COLORS["magspec"], "magspec_controls.bob",
                {}, "Dutch MagSpec: on/off + field"))
            widgets.append(label(name, x - 6, AXIS_Y + 45, 16, 60, rotated=True))
        elif isinstance(elem, Quadrupole):
            if name.startswith("EMQ"):
                widgets.append(button(
                    "", x, AXIS_Y - 45, w, 90, COLORS["emq"], "quad_controls.bob",
                    {"NAME": name}, f"{name}: EMQ current control"))
                widgets.append(label(name, x - 6, AXIS_Y - 105, 16, 56,
                                     rotated=True))
            elif name.startswith("PMQ"):  # fixed magnets on the hexapod
                widgets.append(button(
                    "", x, AXIS_Y - 35, w, 70, COLORS["pmq"],
                    "pmq_hexapod_controls.bob", {},
                    f"{name}: PMQ triplet hexapod (rigid-unit X/Y/Z)"))
                widgets.append(label(name, x - 6, AXIS_Y - 95, 16, 56,
                                     rotated=True))
            else:  # VQ: fixed permanent magnets, no controls
                widgets.append(rect(x, AXIS_Y - 35, w, 70, COLORS["pmq"]))
                widgets.append(label(name, x - 6, AXIS_Y - 95, 16, 56,
                                     rotated=True))
        elif isinstance(elem, Dipole):
            if name.startswith("BEND"):
                widgets.append(button(
                    "", x, AXIS_Y - 30, w, 60, COLORS["bend"], "chicane_controls.bob",
                    {}, f"{name}: chicane R56 control"))
                widgets.append(label(name, x - 6, AXIS_Y + 40, 16, 50,
                                     rotated=True))
            else:  # MagSpec
                widgets.append(rect(x, AXIS_Y - 30, w, 60, COLORS["magspec"]))
                widgets.append(label(name, x - 6, AXIS_Y + 40, 16, 60,
                                     rotated=True))
        elif isinstance(elem, (HorizontalCorrector, VerticalCorrector)):
            base = name[:-1]  # S1H -> S1
            if base in seen_kickers:
                continue
            seen_kickers.add(base)
            # collected separately and appended AFTER the loop: kickers are
            # zero-length and share an x with their neighbor (S1 sits at
            # BEND1's entrance), so they must be drawn last to stay on top
            kicker_widgets.append(button(
                "", x, AXIS_Y - 25, 8, 50, COLORS["kicker"], "steering_controls.bob",
                {"S": base}, f"{base}: H/V steering control"))
            kicker_widgets.append(label(base, x - 4, AXIS_Y + 30, 16, 36, rotated=True))
        elif name == "DriftToUndulator":
            # composite bump knobs act at the exit of this drift
            kicker_widgets.append(button(
                "BMP", x + w - 30, AXIS_Y - 68, 30, 24, (255, 200, 0),
                "visa_bump_controls.bob", {},
                "VISA entrance bumps: position/angle via S3+S4"))
        elif isinstance(elem, Screen):
            active = bool(elem.is_active)
            widgets.append(rect(x, AXIS_Y - 40, 4, 80,
                                COLORS["screen"] if active else (150, 200, 180)))
            widgets.append(label(name, x - 6, AXIS_Y - 130, 16, 86, rotated=True))

    # kickers drawn last = top of z-order (see comment in the loop)
    widgets += kicker_widgets

    # legend
    ly = AXIS_Y + 110
    for i, (key, text) in enumerate([
        ("emq", "EMQ (click)"), ("pmq", "PMQ/fixed"), ("bend", "chicane (click)"),
        ("kicker", "steering (click)"), ("screen", "screen"),
    ]):
        lx = X0 + i * 170
        widgets.append(rect(lx, ly, 14, 14, COLORS[key]))
        widgets.append(label(text, lx + 20, ly, 140, 16))

    # camera navigation tabs (vertical strip; loads only the selected camera)
    from htu.lattice import camera_geometry

    def cam_macros(cam):
        g = camera_geometry(cam)
        hx = g.size_x * g.calibration_m / 2 * 1000
        hy = g.size_y * g.calibration_m / 2 * 1000
        return (f"<CAM>{cam}</CAM><XMIN>{-hx:.2f}</XMIN><XMAX>{hx:.2f}</XMAX>"
                f"<YMIN>{-hy:.2f}</YMIN><YMAX>{hy:.2f}</YMAX>")

    tabs = "".join(f"""
      <tab>
        <name>{cam}</name>
        <file>{viewer}</file>
        <macros>{cam_macros(cam)}</macros>
        <group_name></group_name>
      </tab>""" for cam, viewer in CAMERAS)
    widgets.append(f"""  <widget type="navtabs" version="2.0.0">
    <name>CameraTabs</name>
    <tabs>{tabs}
    </tabs>
    <x>{X0}</x><y>{ly + 40}</y><width>900</width><height>640</height>
    <tab_width>150</tab_width>
    <tab_height>30</tab_height>
  </widget>""")

    body = "\n".join(widgets)
    bob = f"""<?xml version="1.0" encoding="UTF-8"?>
<display version="2.0.0">
  <name>HTU Twin Synoptic</name>
  <width>1500</width>
  <height>{ly + 700}</height>
{body}
</display>
"""
    out = Path(__file__).parent / "htu_synoptic.bob"
    out.write_text(bob)
    print(f"wrote {out} ({total_length:.2f} m, {px_per_m:.0f} px/m)")


if __name__ == "__main__":
    main()
