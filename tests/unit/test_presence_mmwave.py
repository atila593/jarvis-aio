"""mmWave / radar presence awareness (Shelly Presence Gen4 and friends, 9.25.0).

The awareness summary (presence.get_presence_summary) must:
  * recognise the whole presence family — `occupancy` AND `presence` device
    classes — not just `occupancy`;
  * map a sensor to the HA AREA its device is in, instead of string-parsing the
    sensor name (which turns "Presence Gen4 Occupancy" into a bogus room);
  * surface occupant COUNT and ILLUMINANCE per occupied room.
"""
import pytest

from fakes import FakeHass


@pytest.fixture
def presence(load):
    return load("presence")


@pytest.fixture
def area(load, presence, monkeypatch):
    """Map a few entities to HA areas without a real registry: patch the area
    resolvers presence uses (audio_routing.entity_area + _area_room_name)."""
    ar_mod = load("audio_routing")

    def _entity_area(_h, eid):
        if "g4" in eid or "living" in eid:
            return "living_room"
        if "office" in eid:
            return "office"
        return None

    monkeypatch.setattr(ar_mod, "entity_area", _entity_area)
    monkeypatch.setattr(presence, "_area_room_name",
                        lambda _h, aid: {"living_room": "living room",
                                         "office": "office"}.get(aid))
    return presence


def test_presence_device_class_recognised(presence):
    # A sensor reporting device_class 'presence' (not 'occupancy') now counts.
    h = FakeHass()
    h.states.set("binary_sensor.den_presence", "on",
                 device_class="presence", friendly_name="Den Presence")
    rooms = presence.get_presence_summary(h)["rooms"]
    assert "den" in rooms


def test_area_mapping_beats_name_parsing(area):
    # "Presence Gen4 Occupancy" in the Living Room area → room "living room",
    # NOT the bogus name-parsed "gen4".
    h = FakeHass()
    h.states.set("binary_sensor.presenceg4_occupancy", "on",
                 device_class="occupancy", friendly_name="Presence Gen4 Occupancy")
    rooms = area.get_presence_summary(h)["rooms"]
    assert "living room" in rooms
    assert "gen4" not in rooms


def test_name_fallback_when_no_area(presence):
    # No area assigned → fall back to the cleaned sensor name (unchanged behaviour).
    h = FakeHass()
    h.states.set("binary_sensor.kitchen_occupancy", "on",
                 device_class="occupancy", friendly_name="Kitchen Occupancy")
    assert "kitchen" in presence.get_presence_summary(h)["rooms"]


def test_count_and_lux_surfaced(area):
    h = FakeHass()
    h.states.set("binary_sensor.presenceg4_occupancy", "on",
                 device_class="occupancy", friendly_name="Presence Gen4 Occupancy")
    h.states.set("sensor.presenceg4_illuminance", "42",
                 device_class="illuminance", friendly_name="Presence Gen4 Illuminance")
    h.states.set("sensor.presenceg4_target_count", "2",
                 friendly_name="Presence Gen4 Target Count")
    det = area.get_presence_summary(h)["room_details"]["living room"]
    assert det["occupants"] == 2
    assert det["lux"] == 42.0


def test_count_lux_only_for_occupied_rooms(area):
    # A lux sensor in an area with NO active presence must not invent a room.
    h = FakeHass()
    h.states.set("sensor.office_illuminance", "100",
                 device_class="illuminance", friendly_name="Office Illuminance")
    summ = area.get_presence_summary(h)
    assert summ["rooms"] == {}
    assert summ["room_details"] == {}


def test_non_numeric_count_ignored(area):
    h = FakeHass()
    h.states.set("binary_sensor.presenceg4_occupancy", "on",
                 device_class="occupancy", friendly_name="Presence Gen4 Occupancy")
    h.states.set("sensor.presenceg4_target_count", "unavailable",
                 friendly_name="Presence Gen4 Target Count")
    det = area.get_presence_summary(h)["room_details"].get("living room", {})
    assert "occupants" not in det


def test_context_string_mentions_count_and_lux(area):
    h = FakeHass()
    h.states.set("person.sam", "home", friendly_name="Sam")
    h.states.set("binary_sensor.presenceg4_occupancy", "on",
                 device_class="occupancy", friendly_name="Presence Gen4 Occupancy")
    h.states.set("sensor.presenceg4_illuminance", "42",
                 device_class="illuminance", friendly_name="Presence Gen4 Illuminance")
    h.states.set("sensor.presenceg4_target_count", "2",
                 friendly_name="Presence Gen4 Target Count")
    ctx = area.presence_context_string(h)
    assert "living room" in ctx
    assert "2 people" in ctx
    assert "42 lux" in ctx


def test_motion_excluded_from_awareness(presence):
    # Bare motion is too transient for the "who's where" view (audio_routing still
    # uses it for routing).
    h = FakeHass()
    h.states.set("binary_sensor.hall_motion", "on",
                 device_class="motion", friendly_name="Hall Motion")
    assert presence.get_presence_summary(h)["rooms"] == {}
