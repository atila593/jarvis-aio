"""Vocal biometrics Path B — Step 1 (issue #397): the enrolled-voiceprint store
(``voice_roster``) and the speaker matcher (``speaker_id``). Pure, dependency-
free, and inert (no audio capture, no embedding backend) until later steps.

The store keeps embeddings, never audio, as a running-mean centroid per person;
the matcher cosine-matches and only calls a result *trusted* above a conservative
threshold with enough samples — the line a caller must clear before voice may
stand an intrusion alert down."""
import math

import pytest


@pytest.fixture
def vr(load, tmp_path, monkeypatch):
    mod = load("voice_roster")
    monkeypatch.setattr(mod, "VOICE_REF_DIR", str(tmp_path / "voices_ref"))
    return mod


@pytest.fixture
def sid(load):
    return load("speaker_id")


def _vec(dim, hot=0):
    """A simple dim-length vector with a 1.0 in slot `hot`, else 0.1."""
    return [1.0 if i == hot else 0.1 for i in range(dim)]


# ── voice_roster ─────────────────────────────────────────────────────────────

def test_enroll_creates_profile(vr):
    assert vr.enroll("Sam", _vec(32, 0)) is True
    prof = vr.get("Sam")
    assert prof is not None
    assert prof["count"] == 1
    assert prof["dim"] == 32
    assert prof["name"] == "Sam"
    assert vr.has("sam") is True            # normalized lookup
    assert math.isclose(math.sqrt(sum(x * x for x in prof["centroid"])), 1.0, rel_tol=1e-6)


def test_enroll_increments_and_averages(vr):
    vr.enroll("Sam", _vec(32, 0))
    vr.enroll("Sam", _vec(32, 1))
    prof = vr.get("Sam")
    assert prof["count"] == 2
    # centroid is the mean of the two L2-normalized samples (each ~unit length)
    assert 0.0 < math.sqrt(sum(x * x for x in prof["centroid"])) <= 1.0


def test_enroll_rejects_bad_embedding(vr):
    assert vr.enroll("Sam", []) is False
    assert vr.enroll("Sam", "not-a-vector") is False
    assert vr.enroll("Sam", [0.0] * 32) is False      # zero vector → no direction
    assert vr.enroll("Sam", [1.0, 2.0]) is False       # too short (< _MIN_DIM)
    assert vr.get("Sam") is None


def test_enroll_rejects_dim_mismatch(vr):
    assert vr.enroll("Sam", _vec(32, 0)) is True
    assert vr.enroll("Sam", _vec(48, 0)) is False      # different model → reject
    assert vr.get("Sam")["dim"] == 32
    assert vr.get("Sam")["count"] == 1


def test_remove(vr):
    vr.enroll("Sam", _vec(32, 0))
    assert vr.remove("Sam") is True
    assert vr.remove("Sam") is False
    assert vr.get("Sam") is None


def test_profiles_listing(vr):
    vr.enroll("Sam", _vec(32, 0))
    vr.enroll("Jarvis", _vec(32, 1))
    assert vr.profiles() == ["Jarvis", "Sam"]
    assert len(vr.all_profiles()) == 2


def test_name_key_is_path_safe(vr):
    # A hostile name must never escape the voices_ref directory.
    assert vr.enroll("../../etc/passwd", _vec(32, 0)) is True
    import os
    files = os.listdir(vr.VOICE_REF_DIR)
    assert files and all("/" not in f and ".." not in f[:-5] for f in files)


# ── speaker_id matcher ───────────────────────────────────────────────────────

def test_cosine(sid):
    assert math.isclose(sid.cosine([1, 0, 0], [1, 0, 0]), 1.0)
    assert math.isclose(sid.cosine([1, 0], [0, 1]), 0.0)
    assert sid.cosine([1, 0], [1, 0, 0]) == 0.0        # mismatched length
    assert sid.cosine([0, 0], [0, 0]) == 0.0           # zero vector


def test_match_picks_best_and_marks_trusted(sid):
    profs = [
        {"name": "Sam", "centroid": _vec(32, 0), "count": 3},
        {"name": "Jarvis", "centroid": _vec(32, 5), "count": 3},
    ]
    res = sid.match(_vec(32, 0), profs)
    assert res is not None
    assert res["name"] == "Sam"
    assert res["trusted"] is True
    assert res["score"] > 90


def test_match_guess_not_trusted_below_sample_floor(sid):
    # A strong similarity but only one enrolled sample → guess, not trusted.
    profs = [{"name": "Sam", "centroid": _vec(32, 0), "count": 1}]
    res = sid.match(_vec(32, 0), profs, min_samples=2)
    assert res is not None and res["name"] == "Sam"
    assert res["trusted"] is False


def test_match_below_guess_floor_returns_none(sid):
    profs = [{"name": "Sam", "centroid": _vec(32, 0), "count": 3}]
    # An orthogonal embedding is far below the guess floor.
    assert sid.match([0.1] * 31 + [1.0], profs) is None


def test_identify_reads_roster(vr, sid):
    vr.enroll("Sam", _vec(32, 0))
    vr.enroll("Sam", _vec(32, 0))           # 2 samples → eligible to be trusted
    res = sid.identify(_vec(32, 0))
    assert res is not None and res["name"] == "Sam"
    assert res["trusted"] is True


def test_backend_is_inert_without_model(sid):
    # No onnxruntime + model in the sandbox → the whole embed path is a no-op.
    assert sid.backend_available() is False
    assert sid.embed_pcm(b"\x00\x00" * 16000, 16000) is None
    assert sid.embed_wav(b"RIFFxxxx") is None


# ── speaker_id audio helpers (pure, no model needed) ─────────────────────────

def test_pcm_to_float_decodes_16bit_mono(sid):
    import struct
    raw = struct.pack("<4h", 0, 16384, -16384, 32767)
    out = sid._pcm_to_float(raw, 2, 1)
    assert len(out) == 4
    assert out[0] == 0.0
    assert 0.49 < out[1] < 0.51
    assert all(-1.0 <= x <= 1.0 for x in out)


def test_pcm_to_float_downmixes_stereo(sid):
    import struct
    raw = struct.pack("<4h", 100, 300, 0, 0)   # 2 stereo frames
    out = sid._pcm_to_float(raw, 2, 2)
    assert len(out) == 2                         # mono after downmix


def test_resample_to_target_changes_length(sid):
    src = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    out = sid._resample_to_target(src, 8000)     # 8k → 16k ≈ doubles
    assert abs(len(out) - 2 * len(src)) <= 2
    assert sid._resample_to_target(src, 16000) is src   # no-op at target rate


def test_decode_wav_roundtrip(sid):
    import io
    import wave
    import struct
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<8h", *range(8)))
    samples, sr = sid._decode_wav(buf.getvalue())
    assert sr == 16000
    assert len(samples) == 8


# ── voice_recognition in-process provider (Step 3) ───────────────────────────

@pytest.fixture
def vrec(load, monkeypatch):
    mod = load("voice_recognition")
    mod.note_local_speaker(None)
    return mod


def _flag_on(mod, monkeypatch):
    monkeypatch.setattr(mod, "_cfg",
                        lambda k, d: True if k == "voice_recognition_local" else d)


def test_note_and_current_roundtrip(vrec):
    vrec.note_local_speaker({"name": "Sam", "score": 90.0, "trusted": True})
    cur = vrec.current_local_speaker()
    assert cur and cur["name"] == "Sam"


def test_current_speaker_goes_stale(vrec):
    vrec.note_local_speaker({"name": "Sam", "trusted": True, "score": 90})
    vrec._local_speaker["ts"] = 0.0          # force past the recency window
    assert vrec.current_local_speaker() is None


def test_identify_local_off_by_default(vrec, fake_hass):
    vrec.note_local_speaker({"name": "Sam", "trusted": True, "score": 90})
    assert vrec.identify_local(fake_hass) == {}     # flag defaults off


def test_identify_local_trusted_only(vrec, fake_hass, monkeypatch):
    _flag_on(vrec, monkeypatch)
    vrec.note_local_speaker({"name": "Sam", "trusted": True, "score": 88})
    assert vrec.identify_local(fake_hass) == {"Sam": pytest.approx(0.88)}


def test_identify_local_withholds_a_guess(vrec, fake_hass, monkeypatch):
    _flag_on(vrec, monkeypatch)
    vrec.note_local_speaker({"name": "Sam", "trusted": False, "score": 60})
    assert vrec.identify_local(fake_hass) == {}     # a guess never feeds identity


def test_identify_falls_back_to_local_when_no_source(vrec, fake_hass, monkeypatch):
    # No external source configured → identify() uses the in-process trusted vote.
    monkeypatch.setattr(vrec, "_cfg", lambda k, d: {
        "voice_recognition_source": "",
        "voice_recognition_local": True,
    }.get(k, d))
    vrec.note_local_speaker({"name": "Sam", "trusted": True, "score": 91})
    assert vrec.identify(fake_hass) == {"Sam": pytest.approx(0.91)}
