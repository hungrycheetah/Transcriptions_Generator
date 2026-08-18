# Meeting Transcript Generator

Upload a meeting recording, generate a transcript, optionally detect speakers, preview the result, and download the `.txt` file.

## Required Files

- `web_app.py` - FastAPI upload/status/download backend
- `transcribe.py` - Whisper + optional pyannote speaker diarization runner
- `static/` - browser UI
- `requirements.txt` - transcription dependencies
- `requirements-speakers.txt` - speaker diarization dependencies
- `requirements-web.txt` - web server dependencies
- `Dockerfile` - deployable container

## Environment Variables

Speaker detection needs a Hugging Face token with access to `pyannote/speaker-diarization-community-1`.

```text
HF_TOKEN=hf_your_hugging_face_token
TRANSCRIBE_WORKERS=1
TRANSCRIBE_TIMEOUT_SECONDS=7200
```

Do not commit `.env` or `transcripts.env`.

## Local Run

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-speakers.txt
python -m pip install -r requirements-web.txt
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Docker Run

```powershell
docker build -t meeting-transcript-app .
docker run --rm -p 8000:8000 -e HF_TOKEN="hf_your_token" meeting-transcript-app
```

Open:

```text
http://127.0.0.1:8000
```

## Deployment Notes

- Use a Docker-capable host because the app needs FFmpeg and ML dependencies.
- Set `HF_TOKEN` in the host's environment/secrets settings.
- CPU diarization is slow. Longer recordings can take several minutes.
- Uploaded recordings and transcripts are stored under `app_data/`. On ephemeral hosts, files disappear after restart unless persistent disk is configured.
