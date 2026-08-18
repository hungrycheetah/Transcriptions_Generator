import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path


MODEL_CHOICES = ("tiny", "base", "small", "medium", "large")
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio or video file with OpenAI Whisper."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the audio/video file to transcribe.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the transcript file. Defaults to '<input file>.txt'.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=MODEL_CHOICES,
        default="base",
        help="Whisper model to use. Defaults to 'base'.",
    )
    parser.add_argument(
        "--language",
        help="Optional spoken language hint, such as 'en'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the transcript text to the console.",
    )
    parser.add_argument(
        "--no-beep",
        action="store_true",
        help="Skip the completion beep on Windows.",
    )
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="Add speaker labels using pyannote.audio diarization.",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for pyannote diarization. Can also use HF_TOKEN env var.",
    )
    parser.add_argument(
        "--speaker-names",
        help=(
            "Optional names for speakers. Use either 'SPEAKER_00=Name,SPEAKER_01=Name' "
            "or 'Name 1,Name 2'."
        ),
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=f"pyannote diarization model to use. Defaults to '{DEFAULT_DIARIZATION_MODEL}'.",
    )
    return parser.parse_args()


def beep():
    try:
        import winsound

        winsound.Beep(1000, 500)
    except Exception:
        pass


def format_timestamp(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_speaker_names(raw_names):
    if not raw_names:
        return {}

    parts = [part.strip() for part in raw_names.split(",") if part.strip()]
    mapping = {}

    for index, part in enumerate(parts):
        if "=" in part:
            label, name = part.split("=", 1)
            mapping[label.strip()] = name.strip()
        else:
            mapping[f"SPEAKER_{index:02d}"] = part

    return mapping


def load_env_file(env_path):
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def extract_wav_for_diarization(input_path):
    temp_root = Path(__file__).with_name(".cache").joinpath("tmp")
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(
        prefix="transcribe-diarization-",
        dir=temp_root,
        ignore_cleanup_errors=True,
    )
    wav_path = Path(temp_dir.name).joinpath("audio.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-vn",
        str(wav_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        temp_dir.cleanup()
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and make sure it is on PATH.")
    except subprocess.CalledProcessError as error:
        temp_dir.cleanup()
        raise RuntimeError(f"FFmpeg failed to extract audio:\n{error.stderr}") from error

    return temp_dir, wav_path


def get_diarization_turns(input_path, hf_token, diarization_model):
    from pyannote.audio import Pipeline
    from scipy.io import wavfile
    import torch

    print(f"Loading speaker diarization model: {diarization_model}")
    try:
        pipeline = Pipeline.from_pretrained(
            diarization_model,
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            diarization_model,
            use_auth_token=hf_token,
        )

    temp_dir = None
    print(f"[{time.strftime('%H:%M:%S')}] Detecting speakers: {input_path}")
    try:
        temp_dir, wav_path = extract_wav_for_diarization(input_path)
        sample_rate, audio = wavfile.read(str(wav_path))
        waveform = torch.from_numpy(audio).float().unsqueeze(0) / 32768.0
        diarization_output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    finally:
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
            except PermissionError:
                pass

    turns = []
    diarization = getattr(diarization_output, "speaker_diarization", diarization_output)

    if hasattr(diarization, "itertracks"):
        diarization_items = (
            (turn, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        )
    else:
        diarization_items = diarization

    for turn, speaker in diarization_items:
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )

    return turns


def validate_speaker_setup(hf_token):
    if not hf_token:
        print(
            "Speaker diarization needs a Hugging Face token.\n"
            "Pass --hf-token YOUR_TOKEN or set the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        return False

    try:
        import pyannote.audio  # noqa: F401
    except ImportError as error:
        print(
            "Speaker diarization needs pyannote.audio.\n"
            "Install or repair the speaker dependencies with:\n"
            "python -m pip install --force-reinstall -r requirements-speakers.txt\n"
            f"Details: {error}",
            file=sys.stderr,
        )
        return False

    return True


def find_speaker_for_segment(segment, turns):
    segment_start = float(segment["start"])
    segment_end = float(segment["end"])
    overlap_by_speaker = {}

    for turn in turns:
        overlap_start = max(segment_start, turn["start"])
        overlap_end = min(segment_end, turn["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > 0:
            speaker = turn["speaker"]
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if not overlap_by_speaker:
        return "UNKNOWN"

    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def build_speaker_transcript(segments, turns, speaker_names):
    lines = []
    previous_speaker = None

    for segment in segments:
        speaker_label = find_speaker_for_segment(segment, turns)
        speaker_name = speaker_names.get(speaker_label, speaker_label)
        text = segment["text"].strip()

        if not text:
            continue

        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])

        if speaker_name == previous_speaker and lines:
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(f"[{start} - {end}] {speaker_name}: {text}")
            previous_speaker = speaker_name

    return "\n".join(lines)


def main():
    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    load_env_file(Path(__file__).with_name(".env"))
    load_env_file(Path(__file__).with_name("transcripts.env"))
    load_env_file(Path(__file__).with_name(".env.example"))
    matplotlib_cache = Path(__file__).with_name(".cache").joinpath("matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = parse_args()
    input_path = args.input.expanduser()
    output_path = args.output or input_path.with_name(f"{input_path.name}.txt")
    output_path = output_path.expanduser()

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not args.overwrite:
        print(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite or choose a different --output path.",
            file=sys.stderr,
        )
        return 1

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.speakers and not validate_speaker_setup(hf_token):
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import whisper
    except ImportError:
        print(
            "Whisper is not installed. Install dependencies with:\n"
            "python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    print(f"Loading Whisper model: {args.model}")
    try:
        model = whisper.load_model(args.model)
    except Exception as error:
        print(
            f"Failed to load Whisper model '{args.model}'.\n"
            "If this is the first time using this model, Whisper needs internet access "
            "to download it. Try a model that is already cached, or connect to the internet.\n"
            f"Details: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"[{time.strftime('%H:%M:%S')}] Transcribing: {input_path}")
    transcribe_options = {
        "verbose": None if args.quiet else True,
        "fp16": False,
    }
    if args.language:
        transcribe_options["language"] = args.language

    try:
        result = model.transcribe(str(input_path), **transcribe_options)
    except RuntimeError as error:
        print(f"Transcription failed:\n{error}", file=sys.stderr)
        return 1

    if args.speakers:
        try:
            turns = get_diarization_turns(input_path, hf_token, args.diarization_model)
        except Exception as error:
            print(
                "Speaker diarization failed.\n"
                f"Model: {args.diarization_model}\n"
                "If this is a gated Hugging Face model, open its model page, "
                "accept the terms, and make sure your token has read access.\n"
                f"Model page: https://huggingface.co/{args.diarization_model}\n"
                f"Details: {error}",
                file=sys.stderr,
            )
            return 1

        speaker_names = parse_speaker_names(args.speaker_names)
        transcript = build_speaker_transcript(result["segments"], turns, speaker_names)
    else:
        transcript = result["text"].strip()

    if not args.quiet:
        print("\nTranscript:\n")
        print(transcript)

    output_path.write_text(transcript + "\n", encoding="utf-8")
    print(f"\nSaved transcript: {output_path}")

    if not args.no_beep:
        beep()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
