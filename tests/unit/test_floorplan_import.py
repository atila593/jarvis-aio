"""#405 — import a floor plan from a vacuum-robot map / dashboard image entity.

Covers the pure, HA-free logic: which entities are offered as sources, source
enumeration, and the floor_plan_bg merge. The actual image fetch needs a live HA
image/camera entity, so it's exercised only for its defensive shape.
"""
import json

import pytest

from fakes import FakeHass


@pytest.fixture
def fp(load):
    return load("floorplan_import")


# ── discovery heuristic ───────────────────────────────────────────────────────

def test_image_domain_is_always_a_source(fp):
    assert fp.is_floorplan_source("image.cleaner_living_room_map_0") is True
    assert fp.is_floorplan_source("image.anything") is True


def test_camera_needs_a_map_token(fp):
    assert fp.is_floorplan_source("camera.robot_map", "Robot Map") is True
    assert fp.is_floorplan_source("camera.valetudo_map") is True
    assert fp.is_floorplan_source("camera.front_door", "Front Door") is False


def test_non_image_non_camera_rejected(fp):
    assert fp.is_floorplan_source("sensor.cleaner_map") is False
    assert fp.is_floorplan_source("binary_sensor.vacuum_map") is False
    assert fp.is_floorplan_source("") is False


def test_camera_matches_by_friendly_name(fp):
    # id has no token, but the friendly name names a floor plan.
    assert fp.is_floorplan_source("camera.xyz", "Downstairs Floor Plan") is True


# ── source enumeration ────────────────────────────────────────────────────────

def test_list_sources_collects_and_sorts(fp):
    h = FakeHass()
    h.states.set("image.cleaner_map", "idle", friendly_name="Cleaner Map")
    h.states.set("camera.robot_map", "idle", friendly_name="Robot Map")
    h.states.set("camera.front_door", "idle", friendly_name="Front Door")
    h.states.set("sensor.temp", "20", friendly_name="Temp")
    out = fp.list_sources(h)
    ids = [s["entity_id"] for s in out]
    assert "image.cleaner_map" in ids
    assert "camera.robot_map" in ids
    assert "camera.front_door" not in ids   # no map token
    assert "sensor.temp" not in ids          # wrong domain
    # images first, then cameras
    assert out[0]["domain"] == "image"


# ── floor_plan_bg merge ───────────────────────────────────────────────────────

def test_merge_into_empty(fp):
    out = fp._merge_background(None, "1f", "data:image/png;base64,AAA")
    assert json.loads(out) == {"1f": "data:image/png;base64,AAA"}


def test_merge_preserves_other_floors(fp):
    prior = json.dumps({"2f": "data:image/png;base64,OLD"})
    out = fp._merge_background(prior, "1f", "data:image/png;base64,NEW")
    parsed = json.loads(out)
    assert parsed["2f"] == "data:image/png;base64,OLD"
    assert parsed["1f"] == "data:image/png;base64,NEW"


def test_merge_accepts_dict_and_overwrites_floor(fp):
    out = fp._merge_background({"1f": "old"}, "1f", "new")
    assert json.loads(out) == {"1f": "new"}


def test_merge_tolerates_garbage_prior(fp):
    out = fp._merge_background("not json", "1f", "data:image/png;base64,X")
    assert json.loads(out) == {"1f": "data:image/png;base64,X"}
