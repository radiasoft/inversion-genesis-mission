"""Tests for the Phoebus display generators in `docs/htu-twin/phoebus` (no Phoebus needed)."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from lume_fbpic.actions import make_descriptor_actions

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "htu-twin" / "phoebus"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _DOCS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lpa_display():
    return _load("generate_lpa_display")


@pytest.fixture(scope="module")
def linked_synoptic():
    return _load("generate_linked_synoptic")


# A stand-in for the twin's synoptic: a title, an SRC button, and two other buttons.
_SYNOPTIC = """<?xml version="1.0" encoding="UTF-8"?>
<display version="2.0.0">
  <name>HTU Twin Synoptic</name>
  <width>1500</width>
  <height>960</height>
  <widget type="label" version="2.0.0">
    <name>title</name>
    <text>HTU twin — synoptic</text>
    <x>30</x><y>10</y><width>900</width><height>30</height>
  </widget>
  <widget type="action_button" version="3.0.0">
    <name>btn_SRC_2</name>
    <actions>
      <action type="open_display">
        <file>source_controls.bob</file>
        <macros></macros>
        <target>window</target>
        <description>LPA source parameters</description>
      </action>
    </actions>
    <text>SRC</text>
    <x>2</x><y>120</y><width>26</width><height>60</height>
    <tooltip>LPA source parameters</tooltip>
  </widget>
  <widget type="label" version="2.0.0">
    <name>lbl_vertical</name>
    <text>PMQ1V</text>
    <x>31</x><y>55</y><width>16</width><height>56</height>
    <rotation_step>90.0</rotation_step>
  </widget>
  <widget type="tabs" version="2.0.0">
    <name>cameras</name>
    <tabs>
      <tab>
        <name>Phosphor1</name>
        <file>camera_view.bob</file>
        <macros><CAM>Phosphor1</CAM></macros>
      </tab>
    </tabs>
    <x>10</x><y>200</y><width>400</width><height>300</height>
  </widget>
  <widget type="action_button" version="3.0.0">
    <name>btn_EMQ1H_9</name>
    <actions>
      <action type="open_display">
        <file>quad_controls.bob</file>
        <macros><NAME>EMQ1H</NAME></macros>
        <target>window</target>
        <description>EMQ1H</description>
      </action>
    </actions>
    <text>EMQ1H</text>
    <x>434</x><y>120</y><width>30</width><height>60</height>
  </widget>
</display>
"""


@pytest.fixture()
def twin_dir(tmp_path):
    directory = tmp_path / "display"
    directory.mkdir()
    (directory / "htu_synoptic.bob").write_text(_SYNOPTIC, encoding="utf-8")
    return directory


def _files(root):
    return [a.findtext("file") for a in root.iter("action") if a.get("type") == "open_display"]


# --- the LPA bunch display


def test_the_lpa_display_is_valid_xml_with_unique_widget_names(lpa_display):
    root = ET.fromstring(lpa_display.build())

    names = [w.findtext("name") for w in root.findall("widget")]
    assert len(names) == len(set(names))
    assert root.findtext("name") == "LPA bunch"


def test_the_lpa_display_shows_every_descriptor_feature_once(lpa_display):
    root = ET.fromstring(lpa_display.build())
    pvs = [w.findtext("pv_name") for w in root.findall("widget") if w.findtext("pv_name")]

    for action in make_descriptor_actions():
        assert pvs.count(f"pva://$(P){action.name}") == 1, action.name
    assert pvs.count("pva://$(P)LPA_Archive") == 1  # the run selector
    assert len(pvs) == 33 + 14 + 3 + 1  # descriptor, twin source readbacks, statistics, selector


def test_the_lpa_display_prefix_is_a_macro_defaulting_to_the_twins(lpa_display):
    root = ET.fromstring(lpa_display.build())

    assert root.find("macros").findtext("P") == "HTU:SIM:"
    assert all(
        w.findtext("pv_name").startswith("pva://$(P)")
        for w in root.findall("widget")
        if w.findtext("pv_name")
    )


def test_the_lpa_display_links_back_to_the_linked_synoptic(lpa_display):
    root = ET.fromstring(lpa_display.build())

    assert "htu_synoptic_lpa.bob" in _files(root)


def test_the_committed_lpa_display_is_up_to_date(lpa_display):
    committed = (_DOCS / "lpa_bunch.bob").read_text()

    assert committed == lpa_display.build()


# --- the linked synoptic


def test_other_buttons_are_pointed_at_the_twins_display_directory(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    assert "$(TWIN_DISPLAYS)/quad_controls.bob" in _files(root)
    assert root.find("macros").findtext("TWIN_DISPLAYS") == str(twin_dir.resolve())
    quad = [a for a in root.iter("action") if a.findtext("file", "").endswith("quad_controls.bob")][0]
    assert quad.find("macros").findtext("NAME") == "EMQ1H"  # the button's own macros are kept


def test_the_src_button_opens_the_lpa_display_as_its_only_action(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    src = [w for w in root.findall("widget") if w.findtext("name").startswith("btn_SRC")][0]

    # One action: Phoebus draws several as a drop-down menu, which a click would only open.
    assert [a.findtext("file") for a in src.find("actions")] == ["lpa_bunch.bob"]
    assert "LPA bunch" in src.findtext("tooltip")


def test_a_strip_below_the_synoptic_shows_the_source_pvs(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    assert int(root.findtext("height")) == 960 + linked_synoptic.STRIP_HEIGHT
    strip = [w for w in root.findall("widget") if w.findtext("name", "").startswith("lpa_strip")]
    assert min(int(w.findtext("y")) for w in strip) >= 960  # nothing overlaps the original
    pvs = {w.findtext("pv_name") for w in strip if w.findtext("pv_name")}
    assert pvs == {
        "pva://HTU:SIM:LPA_Archive",
        "pva://HTU:SIM:Source_Energy_MeV",
        "pva://HTU:SIM:Source_EnergySpread_pct",
        "pva://HTU:SIM:Source_Charge_pC",
        "pva://HTU:SIM:Source_NumParticles",
    }
    assert "lpa_bunch.bob" in _files(root)  # the strip button opens the LPA display


def test_the_twins_own_synoptic_is_not_modified(linked_synoptic, twin_dir):
    before = (twin_dir / "htu_synoptic.bob").read_bytes()

    linked_synoptic.build(twin_dir)

    assert (twin_dir / "htu_synoptic.bob").read_bytes() == before


def test_a_synoptic_without_one_src_button_is_refused(linked_synoptic, twin_dir):
    text = (twin_dir / "htu_synoptic.bob").read_text(encoding="utf-8").replace("btn_SRC_2", "btn_X")
    (twin_dir / "htu_synoptic.bob").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="SRC button"):
        linked_synoptic.build(twin_dir)


def test_non_ascii_text_in_the_twins_synoptic_survives(linked_synoptic, twin_dir):
    assert "—" in linked_synoptic.build(twin_dir)


def test_embedded_camera_tabs_are_pointed_at_the_twins_directory_too(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    tab_files = [t.findtext("file") for t in root.iter("tab")]

    assert tab_files == ["$(TWIN_DISPLAYS)/camera_view.bob"]
    assert root.find(".//tab/macros").findtext("CAM") == "Phosphor1"  # the tab's macros are kept


def test_no_display_file_is_left_relative_except_the_lpa_display(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    relative = {f.text for f in root.iter("file") if not f.text.startswith("$(TWIN_DISPLAYS)/")}

    assert relative == {"lpa_bunch.bob"}
    assert "source_controls.bob" not in " ".join(f.text for f in root.iter("file"))  # replaced


def test_the_run_selector_is_a_combo_box_that_takes_its_items_from_the_pv(lpa_display, linked_synoptic, twin_dir):
    for xml in (lpa_display.build(), linked_synoptic.build(twin_dir)):
        root = ET.fromstring(xml)

        combos = [w for w in root.findall("widget") if w.get("type") == "combo"]

        assert len(combos) == 1
        assert combos[0].findtext("pv_name").endswith("LPA_Archive")
        assert combos[0].findtext("items_from_pv") == "true"


def test_rotation_steps_are_converted_from_degrees_to_phoebus_ordinals(linked_synoptic, twin_dir):
    root = ET.fromstring(linked_synoptic.build(twin_dir))

    steps = [s.text for s in root.iter("rotation_step")]

    assert steps == ["1"]  # 90.0 degrees -> ordinal 1 (NINETY)


def test_a_rotation_that_is_not_a_multiple_of_90_degrees_is_refused(linked_synoptic, twin_dir):
    text = (twin_dir / "htu_synoptic.bob").read_text(encoding="utf-8").replace("90.0", "45.0")
    (twin_dir / "htu_synoptic.bob").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="multiple of 90"):
        linked_synoptic.build(twin_dir)
