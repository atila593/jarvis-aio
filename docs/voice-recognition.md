# JARVIS Voice Recognition — Setup

JARVIS can know who it's talking to by **voice**, fuse that with who's home and
who's on camera, and **learn people's voices over time** from ordinary
conversation. This is the setup guide.

## How it's put together

JARVIS does **not** run a speaker-embedding model inside Home Assistant — that's
heavy, dependency-laden, and needs the raw utterance audio. Instead a dedicated
speaker-recognition service does the embedding + enrollment and publishes *who is
speaking* to Home Assistant; JARVIS **consumes** that signal and feeds it into
its identity resolver's voice tier (the strongest tier).

```
  Voice satellite / Assist audio
            │
            ▼
  Speaker-recognition service          ← does the ML (embeddings, enrollment)
  (VoiceBM · speaker-recognition · …)
            │  publishes "current speaker" to HA
            ▼
  binary_sensor.<person>_voice  /  sensor.current_speaker
            │
            ▼
  JARVIS identity resolver  ──►  voice + presence + face  ──►  who you are
```

You supply the service; JARVIS supplies the fusion and the personalization
(per-person memory, routines, command attribution) it already has.

## Step 1 — Run a speaker-recognition service

Any backend works as long as it surfaces the current speaker as an HA entity.
Two good options:

- **VoiceBM** — `github.com/cybericebyte/VoiceBM`. Sherpa-ONNX, **CPU-only**,
  publishes over MQTT with HA discovery. Gives you a per-person
  `binary_sensor.<person>_voice` (ON while they speak) and a current-speaker
  sensor, plus a built-in enrollment/review flow. Closest fit.
- **speaker-recognition** — `github.com/EuleMitKeule/speaker-recognition`.
  Resemblyzer, runs as a Home Assistant **add-on** + integration exposing a
  current-speaker sensor, with `/train` and `/recognize` REST endpoints.

Install and enroll a couple of voices per their instructions, then confirm the
entity exists in **Developer Tools → States** (e.g. `binary_sensor.sam_voice`
flips ON when Sam speaks, or `sensor.current_speaker` shows a name).

> GPU note: neither service *requires* a GPU (both default to CPU/ONNX). Your
> Ollama GPU is free to keep serving the LLM. If you later want faster or
> higher-accuracy embeddings, point the service's model at the GPU — that's a
> service-side change, independent of JARVIS.

## Step 2 — Point JARVIS at it

In **Settings → Devices & Services → JARVIS → Configure → Identity**:

- **Enable the voice tier** — turn on *voice fingerprint* (`identity_voice_fingerprint`).
- **Voice recognition source** (`voice_recognition_source`) — one of:
  - a glob for per-person sensors: `binary_sensor.*_voice` (VoiceBM style), or
  - a single current-speaker sensor: `sensor.current_speaker`.

That's it — JARVIS now weights voice most heavily, and falls back to who's home /
who's on camera when the voice is uncertain, so it degrades gracefully.

**Confidence:** if your source publishes a confidence/score attribute JARVIS uses
it (accepts 0–1 or 0–100); otherwise it uses `voice_recognition_confidence`
(default 0.85).

## Step 3 — Learn voices over time (hands-free enrollment)

The service does the enrolling, but the hard part is the **label** — *who* is this
unknown voice? JARVIS already knows the answer when you're the only one home or a
camera just recognized your face. So when JARVIS is confident who's speaking from
those signals but the voice service doesn't recognize the voice yet, it fires:

```
event: jarvis_voice_enroll_candidate
data:  { person: "sam", device_id: "…" }
```

Wire that to your service's enrollment so profiles build themselves from normal
conversation. Example (VoiceBM, enrolling the pending sample under the named
person):

```yaml
automation:
  - alias: "JARVIS · auto-enroll voice"
    trigger:
      - platform: event
        event_type: jarvis_voice_enroll_candidate
    action:
      # Enroll VoiceBM's pending utterance under the person JARVIS identified.
      # (Use your service's actual enroll service / MQTT command.)
      - service: mqtt.publish
        data:
          topic: "voicebm/enroll"
          payload: "{{ trigger.event.data.person }}"
```

For `speaker-recognition`, call its `/train` with the pending sample tagged with
`trigger.event.data.person` instead.

Auto-flagging is rate-limited (once per person per 5 minutes) and can be turned
off with **Auto-flag voices to enroll** (`voice_recognition_auto_enroll`).

## Alternative: JARVIS's own in-process speaker ID (experimental, no external service)

If you don't want to run a separate speaker-recognition service, JARVIS can do
the biometrics **itself** — the voice analog of the experimental LLM face
matcher. It's **opt-in, off by default, and fail-open**: nothing captures audio,
loads a model, or changes identity until you enable it, and if anything is
missing the whole path simply stays inert. Voiceprints are stored as
**embeddings, never audio**.

> This path is more involved than the external service above and needs a model
> you supply. If you want the easiest reliable route, use Step 1–3. Use this when
> you'd rather keep everything inside Home Assistant.

**How it works:** JARVIS registers an optional **STT "tee"**. When you select it
as your Assist pipeline's speech-to-text, it passes the audio straight through to
a *delegate* STT (Whisper/cloud) for the real transcription — adding no latency —
and keeps a copy to turn into a voiceprint in the background. The result only
identifies a resident above a confidence bar; a weaker match is shown as a *guess*
and **never** stands a security/intrusion alert down.

### Step A — Install the embedding runtime

Add `onnxruntime` to the Python environment Home Assistant runs in (it is **not**
bundled, to keep default installs light). For HAOS/Supervised this means a custom
build or a venv that includes it; for a Core/venv install:

```bash
pip install onnxruntime
```

Restart Home Assistant afterwards.

### Step B — Supply a speaker-embedding model

Drop an ONNX speaker-embedding model at:

```
<config>/jarvis/models/speaker.onnx
```

(or set a different path in the Voices card — see Step D). **Model contract:** the
default front-end feeds the model **16 kHz mono waveform** and reads back one
embedding vector, which matches *raw-waveform* speaker models (e.g. several
WeSpeaker / 3D-Speaker / ECAPA exports and Resemblyzer-style encoders). A
**feature-based** model that expects pre-computed fbank/MFCC features will not
work with the built-in front-end — those need a model-specific preprocessor,
which is left as a seam in `speaker_id.py`. Pick a raw-waveform export and you're
fine.

JARVIS enrolls and matches with the **same** model, so whatever you choose is
internally consistent even if its absolute scores differ from another model's.

### Step C — Choose a delegate STT

The tee still needs something to actually transcribe. Note the entity id of your
existing speech-to-text (e.g. `stt.faster_whisper`) from **Developer Tools →
States** — you'll point JARVIS at it in the next step.

### Step D — Turn it on (JARVIS panel → Faces → Voices)

Open the JARVIS panel, go to the **Faces** tab, and find the **Voices** card:

1. Toggle **Identify speakers in-process** on.
2. Set **Delegate STT entity** to the id from Step C (e.g. `stt.faster_whisper`).
3. Leave **Model path** blank to use the default, or set your own.

The card shows **● speaker-ID backend ready** once `onnxruntime` + the model are
both found. If it shows **○ backend not installed**, re-check Steps A–B.

### Step E — Select the tee in your Assist pipeline

In **Settings → Voice assistants → (your pipeline) → Speech-to-text**, choose
**JARVIS Speaker-ID**. It delegates transcription to the STT you set in Step C,
so your words are transcribed exactly as before — JARVIS just also listens for
*who* said them. (This entity only appears once Step D is on, a delegate is set,
and the backend is ready.)

### Step F — Enroll voices

In the **Voices** card, type a resident's name, click **🎤 Record & enroll**, and
speak for a few seconds. Repeat a couple of times per person (≥2 samples is the
default bar for a *trusted* match). Use **🎤 Identify** to test — it records a clip
and tells you who JARVIS thinks it is, and whether that's a confident match or a
guess. Hands-free enrollment (the `jarvis_voice_enroll_candidate` event in Step 3)
works here too.

### Trusted vs. guess (the safety gate)

A match becomes **trusted** — and may feed the identity voice tier and stand an
intrusion alert down — only when cosine similarity ≥ `voice_local_trust_threshold`
**and** the person has ≥ `voice_local_min_samples` enrolled samples. Anything from
`voice_local_guess_threshold` up to that bar is a **guess**: shown in the panel,
useful context, but never trusted for security. An external
`voice_recognition_source` (Step 1–3), if configured, always takes precedence over
the in-process result.

### In-process config reference

| Key | Default | Meaning |
|---|---|---|
| `voice_recognition_local` | `false` | Master enable for in-process speaker ID |
| `voice_recognition_delegate_stt` | — | STT entity the tee delegates transcription to (required) |
| `voice_embedding_model_path` | `<config>/jarvis/models/speaker.onnx` | ONNX speaker model (raw-waveform) |
| `voice_local_trust_threshold` | `0.75` | Cosine ≥ this (with enough samples) = trusted |
| `voice_local_guess_threshold` | `0.55` | Cosine ≥ this = panel-only guess |
| `voice_local_min_samples` | `2` | Enrolled samples required before a match can be trusted |

### Troubleshooting

- **Backend shows "not installed"** — `onnxruntime` isn't importable, or no model
  at the path. Install the runtime (Step A), confirm the file exists (Step B),
  restart HA.
- **"JARVIS Speaker-ID" isn't in the STT list** — it only registers when local
  voice ID is on, a delegate STT is set, and the backend is ready (Steps A–D).
- **Never a confident match** — enroll more samples per person, or lower
  `voice_local_trust_threshold` slightly; very short commands ("yes") carry too
  little voice to identify, so the clip needs ~1s+ of speech.
- **It transcribes but never identifies** — the embed runs in the background after
  transcription; check the HA log for `speaker` lines. A `unavailable`/missing
  model logs a one-time warning and stays inert (your voice control is unaffected).

## Config reference

| Key | Default | Meaning |
|---|---|---|
| `identity_voice_fingerprint` | `false` | Master enable for the voice tier |
| `voice_recognition_source` | — | Current-speaker sensor id, or a glob like `binary_sensor.*_voice` |
| `voice_recognition_confidence` | `0.85` | Fallback score when the source carries none |
| `voice_recognition_auto_enroll` | `true` | Fire `jarvis_voice_enroll_candidate` for hands-free learning |
| `voice_recognition_local` | `false` | Use JARVIS's own in-process speaker ID (see the section above) |
| `identity_min_confidence` | `0.45` | Below this, JARVIS says "unknown" rather than guess |

## How JARVIS uses the result

Once JARVIS knows who's speaking, everything per-person it already does kicks in:
commands are attributed to the right person, *their* preferences and facts surface
(and stay private from other residents), and the routines it learns are filed
under the right person. Voice just makes that attribution far more reliable than
presence + face alone.
