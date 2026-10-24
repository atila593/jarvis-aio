"""Echo / Alexa Media Player speakers are routed through Chime TTS.

tts.speak hands the speaker a media-source URL, which Cast/Sonos fetch but an
Echo does not (Alexa plays audio only via the Amazon cloud). When Chime TTS is
installed, alexa_media players are spoken to with ``chime_tts.say`` instead —
same TTS engine, an Alexa-compatible mp3 — while every other speaker keeps the
unchanged tts.speak path. Without Chime TTS (or with ``echo_tts_mode: off``)
nothing changes at all.
"""
import types

import pytest

DOMAIN = "jarvis"
ECHO = "media_player.salon"
ECHO2 = "media_player.cuisine"
CAST = "media_player.kitchen_nest"


@pytest.fixture
def tts(load):
    return load("tts_helper")


@pytest.fixture
def registry(monkeypatch):
    """Entity registry where the *salon*/*cuisine* players are alexa_media."""
    import sys
    er = sys.modules["homeassistant.helpers.entity_registry"]
    platforms = {ECHO: "alexa_media", ECHO2: "alexa_media", CAST: "cast"}
    reg = types.SimpleNamespace(
        async_get=lambda eid: (types.SimpleNamespace(platform=platforms[eid])
                               if eid in platforms else None))
    monkeypatch.setattr(er, "async_get", lambda hass: reg, raising=False)
    return reg


def _chime(hass, installed=True):
    hass.services.has_service = lambda d, s: installed and (d, s) == ("chime_tts", "say")


def _cfg(hass, **kw):
    hass.data.setdefault(DOMAIN, {}).setdefault("e1", {"runtime_config": {}})
    hass.data[DOMAIN]["e1"].setdefault("runtime_config", {}).update(kw)


def _calls(hass, domain, service):
    return [c[2] for c in hass.service_calls if c[0] == domain and c[1] == service]


async def test_echo_goes_through_chime_and_others_stay_on_tts_speak(tts, fake_hass, registry):
    _chime(fake_hass)
    ok = await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO, CAST])
    assert ok is True
    chime = _calls(fake_hass, "chime_tts", "say")
    speak = _calls(fake_hass, "tts", "speak")
    assert len(chime) == 1 and chime[0]["entity_id"] == [ECHO]
    assert chime[0]["tts_platform"] == "tts.piper" and chime[0]["message"] == "hello"
    assert len(speak) == 1 and speak[0]["media_player_entity_id"] == [CAST]


async def test_only_echo_players_never_calls_tts_speak(tts, fake_hass, registry):
    _chime(fake_hass)
    ok = await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO, ECHO2])
    assert ok is True
    assert _calls(fake_hass, "tts", "speak") == []
    assert _calls(fake_hass, "chime_tts", "say")[0]["entity_id"] == [ECHO, ECHO2]


async def test_without_chime_tts_nothing_changes(tts, fake_hass, registry):
    _chime(fake_hass, installed=False)
    await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO, CAST])
    assert _calls(fake_hass, "chime_tts", "say") == []
    assert _calls(fake_hass, "tts", "speak")[0]["media_player_entity_id"] == [ECHO, CAST]


async def test_mode_off_keeps_tts_speak_for_everyone(tts, fake_hass, registry):
    _chime(fake_hass)
    _cfg(fake_hass, echo_tts_mode="off")
    await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO, CAST])
    assert _calls(fake_hass, "chime_tts", "say") == []
    assert _calls(fake_hass, "tts", "speak")[0]["media_player_entity_id"] == [ECHO, CAST]


async def test_echo_voice_and_language_overrides_apply_to_echo_only(tts, fake_hass, registry):
    _chime(fake_hass)
    _cfg(fake_hass, echo_tts_voice="fr_FR-tom-medium", echo_tts_language="fr_FR")
    await tts.async_announce(fake_hass, "bonjour", "tts.piper", [ECHO, CAST])
    chime = _calls(fake_hass, "chime_tts", "say")[0]
    assert chime["options"]["voice"] == "fr_FR-tom-medium"
    assert chime["language"] == "fr_FR"
    # the other speaker keeps the regular JARVIS voice, untouched
    speak = _calls(fake_hass, "tts", "speak")[0]
    assert speak["options"]["voice"] == "en_GB-jarvis-medium"
    assert "language" not in speak


async def test_without_overrides_echo_uses_the_regular_jarvis_voice(tts, fake_hass, registry):
    _chime(fake_hass)
    await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO])
    chime = _calls(fake_hass, "chime_tts", "say")[0]
    assert chime["options"]["voice"] == "en_GB-jarvis-medium"
    assert "language" not in chime


async def test_a_bad_echo_does_not_silence_the_others(tts, fake_hass, registry):
    _chime(fake_hass)
    seen = []

    async def flaky(domain, service, data=None, blocking=False, **kw):
        seen.append((domain, service, dict(data or {})))
        ids = data.get("entity_id") or []
        if domain == "chime_tts" and len(ids) > 1:
            raise RuntimeError("batch rejected")
        if domain == "chime_tts" and ids == [ECHO]:
            raise RuntimeError("echo offline")

    fake_hass.services.async_call = flaky
    ok = await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO, ECHO2])
    assert ok is True
    singles = [c[2]["entity_id"] for c in seen if c[0] == "chime_tts" and len(c[2]["entity_id"]) == 1]
    assert [ECHO2] in singles


async def test_registry_failure_falls_back_to_tts_speak(tts, fake_hass, monkeypatch):
    import sys
    _chime(fake_hass)
    er = sys.modules["homeassistant.helpers.entity_registry"]
    monkeypatch.setattr(er, "async_get", lambda hass: (_ for _ in ()).throw(RuntimeError("boom")), raising=False)
    ok = await tts.async_announce(fake_hass, "hello", "tts.piper", [ECHO])
    assert ok is True
    assert _calls(fake_hass, "chime_tts", "say") == []
    assert _calls(fake_hass, "tts", "speak")[0]["media_player_entity_id"] == [ECHO]
