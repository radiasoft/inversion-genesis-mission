"""Generate `lpa_bunch.bob`, a Phoebus display of the LPA bunch served by `lume-fbpic-serve --twin`.

The display shows the twin's `Source_*` readbacks (the LPA bunch as injected),
the 33 moment-descriptor scalars (the covariance as an upper-triangular grid) and the run
statistics. Every PV is `pva://$(P)<name>`; `P` is a display macro, `HTU:SIM:` by default, the
prefix `lume-fbpic-serve --twin` uses. Open it with a different prefix by setting the macro
(`phoebus -resource lpa_bunch.bob?P=OTHER:`).

A combo box selects among the archives `lume-fbpic-serve` was given (the `LPA_Archive` PV).
A button opens `htu_synoptic_lpa.bob`, the twin's synoptic with the LPA tied in, which
`generate_linked_synoptic.py` writes next to this file.

The PV names come from `lume_fbpic.actions.make_descriptor_actions()`, so the display follows the
code. Run `python generate_lpa_display.py [output.bob]` to regenerate it; do not edit the `.bob`
by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

from lume_fbpic.actions import make_descriptor_actions

SELECTOR = "LPA_Archive"  # the enum PV `lume-fbpic-serve` serves; its options are the archives

COORDS = ("x", "ux", "y", "uy", "z", "uz")
COORD_LABELS = ("x [m]", "ux", "y [m]", "uy", "z [m]", "uz")
PREFIX = "descriptor_"

# (PV name, label) of the twin's source readbacks, in two columns.
SOURCE = (
    ("Source_Energy_MeV", "Energy [MeV]"),
    ("Source_EnergySpread_pct", "Energy spread [% rms]"),
    ("Source_Charge_pC", "Charge [pC]"),
    ("Source_NumParticles", "Macroparticles"),
    ("Source_BetaX_mm", "Beta x [mm]"),
    ("Source_BetaY_mm", "Beta y [mm]"),
    ("Source_AlphaX", "Alpha x"),
    ("Source_AlphaY", "Alpha y"),
    ("Source_NormEmitX_um", "Norm. emit. x [um]"),
    ("Source_NormEmitY_um", "Norm. emit. y [um]"),
    ("Source_X_um", "Centroid x [um]"),
    ("Source_Y_um", "Centroid y [um]"),
    ("Source_Xp_mrad", "Pointing x [mrad]"),
    ("Source_Yp_mrad", "Pointing y [mrad]"),
)
STATS = (
    ("charge_pc", "Charge, all electrons [pC]"),
    ("energy_mean_mev", "Mean energy [MeV]"),
    ("energy_std_mev", "Energy rms [MeV]"),
)


class _Display:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self._widgets: list[str] = []
        self._names: set[str] = set()

    def _unique(self, name: str) -> str:
        base, index = name, 2
        while name in self._names:
            name, index = f"{base}{index}", index + 1
        self._names.add(name)
        return name

    def label(self, name, text, x, y, w, h=22, bold=False, size=None, align=None, wrap=False):
        font = ""
        if bold or size:
            font = (
                f'\n    <font><font family="Liberation Sans" style="{"BOLD" if bold else "REGULAR"}" '
                f'size="{size or 12.0}"/></font>'
            )
        extra = ""
        if align is not None:
            extra += f"\n    <horizontal_alignment>{align}</horizontal_alignment>"
        if wrap:
            extra += "\n    <wrap_words>true</wrap_words>"
        self._widgets.append(
            f'  <widget type="label" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <text>{escape(text)}</text>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>{font}{extra}\n"
            f"  </widget>"
        )

    def value(self, name, pv, x, y, w, h=22, exponential=False, precision=None):
        fmt = ""
        if exponential:
            fmt += "\n    <format>2</format>"
        if precision is not None:
            fmt += f"\n    <precision>{precision}</precision>\n    <precision_from_pv>false</precision_from_pv>"
        self._widgets.append(
            f'  <widget type="textupdate" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <pv_name>pva://$(P){pv}</pv_name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>{fmt}\n"
            f"    <show_units>false</show_units>\n"
            f"  </widget>"
        )

    def button(self, name, text, display, description, x, y, w, h=26):
        self._widgets.append(
            f'  <widget type="action_button" version="3.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <actions>\n"
            f'      <action type="open_display">\n'
            f"        <file>{escape(display)}</file>\n"
            f"        <macros></macros>\n"
            f"        <target>window</target>\n"
            f"        <description>{escape(description)}</description>\n"
            f"      </action>\n"
            f"    </actions>\n"
            f"    <text>{escape(text)}</text>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"  </widget>"
        )

    def combo(self, name, pv, x, y, w, h=24):
        self._widgets.append(
            f'  <widget type="combo" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <pv_name>pva://$(P){pv}</pv_name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>{h}</height>\n"
            f"    <items_from_pv>true</items_from_pv>\n"
            f"  </widget>"
        )

    def rule(self, name, x, y, w):
        self._widgets.append(
            f'  <widget type="rectangle" version="2.0.0">\n'
            f"    <name>{self._unique(name)}</name>\n"
            f"    <x>{x}</x><y>{y}</y><width>{w}</width><height>1</height>\n"
            f'    <line_width>0</line_width>\n'
            f'    <background_color><color red="150" green="150" blue="150"/></background_color>\n'
            f"  </widget>"
        )

    def xml(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<display version="2.0.0">\n'
            "  <name>LPA bunch</name>\n"
            "  <macros>\n    <P>HTU:SIM:</P>\n  </macros>\n"
            f"  <width>{self.width}</width>\n  <height>{self.height}</height>\n"
            + "\n".join(self._widgets)
            + "\n</display>\n"
        )


def build() -> str:
    """The display's XML."""
    features = [a.name[len(PREFIX) :] for a in make_descriptor_actions()]
    bins = sorted(f for f in features if f.startswith("longitudinal_mean_uz_"))
    left, grid_label_w, cell_w, cell_h = 20, 70, 108, 22
    width = left + grid_label_w + 6 * cell_w + 20
    d = _Display(width, 700)

    d.label("Title", "LPA bunch", left, 10, 300, 28, bold=True, size=16.0)
    d.label(
        "Subtitle",
        "Served by lume-fbpic-serve --twin. The values are the selected run's recorded results.",
        left,
        40,
        width - 2 * left,
        20,
        size=10.0,
    )

    d.button(
        "SynopticButton",
        "HTU synoptic...",
        "htu_synoptic_lpa.bob",
        "The HTU twin synoptic with the LPA bunch tied in (see generate_linked_synoptic.py)",
        width - left - 150,
        12,
        150,
    )

    d.label("RunLabel", "LPA run:", left, 70, 70, 24, bold=True)
    d.combo("RunSelector", SELECTOR, left + 74, 70, 260)
    d.label(
        "RunNote",
        "Choose a run: its values are loaded here and passed on to the twin as its source.",
        left + 344,
        70,
        width - left - 344 - left,
        24,
        size=10.0,
    )

    y = 108
    d.label("SourceHeader", "Twin source (the LPA bunch as injected)", left, y, 400, 22, bold=True)
    d.rule("SourceRule", left, y + 24, width - 2 * left)
    y += 32
    column_w = (width - 2 * left) // 2
    for index, (pv, text) in enumerate(SOURCE):
        column, row = index % 2, index // 2
        x = left + column * column_w
        d.label(f"SrcL{index}", text, x, y + row * 26, 150)
        d.value(f"SrcV{index}", pv, x + 155, y + row * 26, 110, precision=4)
    y += 26 * ((len(SOURCE) + 1) // 2) + 14

    d.label("MomentHeader", "Moment descriptor: momentum centroids and charge", left, y, 500, 22, bold=True)
    d.rule("MomentRule", left, y + 24, width - 2 * left)
    y += 32
    for index, name in enumerate(("mean_ux", "mean_uy", "mean_uz")):
        x = left + index * 2 * cell_w
        d.label(f"MeanL{index}", name.replace("mean_", "mean "), x, y, 60)
        d.value(f"MeanV{index}", PREFIX + name, x + 62, y, cell_w, precision=4)
    d.label("ChargeL", "Charge [C]", left + 6 * cell_w - cell_w, y + 26, 90, align=2)
    d.value("ChargeV", PREFIX + "total_beam_charge_c", left + 6 * cell_w + 2, y + 26, cell_w, exponential=True, precision=4)
    y += 62

    d.label(
        "CovHeader",
        "Covariance of (x, ux, y, uy, z, uz); u = p / (m c)",
        left,
        y,
        500,
        22,
        bold=True,
    )
    d.rule("CovRule", left, y + 24, width - 2 * left)
    y += 32
    for column, text in enumerate(COORD_LABELS):
        d.label(f"CovCol{column}", text, left + grid_label_w + column * cell_w, y, cell_w - 4, 20, bold=True, align=1)
    y += 24
    for row, row_text in enumerate(COORD_LABELS):
        d.label(f"CovRow{row}", row_text, left, y + row * (cell_h + 2), grid_label_w - 4, cell_h, bold=True)
        for column in range(row, 6):
            d.value(
                f"Cov_{COORDS[row]}_{COORDS[column]}",
                f"{PREFIX}cov_{COORDS[row]}_{COORDS[column]}",
                left + grid_label_w + column * cell_w,
                y + row * (cell_h + 2),
                cell_w - 4,
                cell_h,
                exponential=True,
                precision=3,
            )
    y += 6 * (cell_h + 2) + 14

    d.label("SliceHeader", "Longitudinal slices (equal particle count along z, tail to head)", left, y, 560, 22, bold=True)
    d.rule("SliceRule", left, y + 24, width - 2 * left)
    y += 32
    d.label("SliceMeanL", "mean uz", left, y + 24, grid_label_w - 4)
    d.label("SliceRmsL", "rms uz", left, y + 50, grid_label_w - 4)
    for index, mean_name in enumerate(bins):
        x = left + grid_label_w + index * cell_w
        d.label(f"SliceCol{index}", f"slice {index}", x, y, cell_w - 4, 20, bold=True, align=1)
        d.value(f"SliceMean{index}", PREFIX + mean_name, x, y + 24, cell_w - 4, precision=4)
        d.value(f"SliceRms{index}", PREFIX + mean_name.replace("mean", "rms"), x, y + 50, cell_w - 4, precision=4)
    y += 84

    d.label("StatsHeader", "Run statistics (blank/NaN for a reconstructed run: no particles)", left, y, 560, 22, bold=True)
    d.rule("StatsRule", left, y + 24, width - 2 * left)
    y += 32
    for index, (pv, text) in enumerate(STATS):
        x = left + index * ((width - 2 * left) // 3)
        d.label(f"StatL{index}", text, x, y, 175, wrap=False)
        d.value(f"StatV{index}", pv, x, y + 24, 140, precision=4)
    y += 60
    d.height = y
    return d.xml()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    output = Path(argv[0]) if argv else Path(__file__).with_name("lpa_bunch.bob")
    output.write_text(build())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
