# Arabic Speech-to-Text Pipeline

100% local, free, open-source Arabic speech-to-text batch transcription using
**whisper.cpp**, **FFmpeg**, and a small Python orchestrator.

No cloud APIs, no external transcription services, no uploads — every file is
processed on this machine.

## How it works

```text
Media file (input/)
    ↓ FFmpeg  (16 kHz mono PCM WAV)
    ↓ whisper.cpp  (language forced to Arabic)
    ↓ plain-text transcript
    ↓
text/<same-name>.txt
```

## Project structure

```text
.
├── input/            ← drop media files here
├── text/             ← transcripts are written here
├── whisper.cpp/      ← cloned + compiled engine (build/bin/whisper-cli)
│   └── models/ggml-small.bin
└── transcribe.py     ← the batch script
```

## Prerequisites

```bash
sudo apt install -y git cmake build-essential ffmpeg python3
```

The engine and model are already built and downloaded in this repo. To rebuild
from scratch on a fresh machine, follow [Setup from scratch](#setup-from-scratch).

## Usage

```bash
python3 transcribe.py
```

1. Put video/audio files (or subfolders of them) in `input/`.
2. Run the script.
3. Read the Arabic transcripts in `text/` — same name, `.txt` extension,
   subdirectories preserved.

The script is restartable: files that already have a transcript in `text/` are
skipped. Re-running only processes new files.

### Options

```bash
python3 transcribe.py \
    --input input \
    --output text \
    --model whisper.cpp/models/ggml-small.bin \
    --language ar \
    --overwrite
```

| Flag          | Default                                     | Purpose                                    |
|---------------|---------------------------------------------|--------------------------------------------|
| `--input`     | `input`                                     | Source media directory (searched recursively) |
| `--output`    | `text`                                      | Transcript destination directory           |
| `--model`     | `whisper.cpp/models/ggml-small.bin`         | GGML Whisper model file                    |
| `--language`  | `ar`                                        | Whisper language code                      |
| `--overwrite` | off                                         | Re-transcribe files that already have a transcript |

Example output:

```text
Scanning input/...
Found 37 media files.

[1/37] lecture01.mp4
    Extracting audio...
    Transcribing Arabic...
    Saved: text/lecture01.txt
...

Finished.
Processed: 34
Skipped:   2
Failed:    1
```

## Supported formats

```
.mp4 .mkv .mov .avi .webm .mp3 .wav .m4a .flac .ogg .opus
```

Any format FFmpeg can decode works; the audio is normalized to 16 kHz mono PCM
before transcription.

## Models

| Model             | Size    | Notes                                        |
|-------------------|---------|----------------------------------------------|
| `small`           | ~466 MB | Default here — good speed/accuracy, fits this machine |
| `large-v3-turbo`  | ~1.6 GB | Recommended default for best accuracy/speed balance |
| `large-v3`        | ~3 GB   | Maximum accuracy                            |

Do **not** use `.en` (English-only) models — the target language is Arabic.

### Switching to a larger model

```bash
cd whisper.cpp
./models/download-ggml-model.sh large-v3-turbo
cd ..

python3 transcribe.py --model whisper.cpp/models/ggml-large-v3-turbo.bin
```

The default model is only used when `--model` is omitted.

## Setup from scratch

```bash
# 1. Clone and build the engine
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build --config Release -j
cd ..

# 2. Download a model (never an English-only one)
cd whisper.cpp
./models/download-ggml-model.sh small
cd ..

# 3. Transcribe
mkdir -p input text
python3 transcribe.py
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `whisper-cli not found` | Engine not built — run the build steps in [Setup from scratch](#setup-from-scratch) |
| `model not found` | Run `./models/download-ggml-model.sh small` in `whisper.cpp/` or pass `--model` |
| `ffmpeg not found` | `sudo apt install ffmpeg` |
| No disk space | Models are large (~0.5–3 GB). Free space or store the model on another disk and pass `--model` |
| Wrong/garbled transcript | Audio may not be Arabic, or use a larger model (`large-v3-turbo`) for accuracy |

## Known limitations

- Whisper does not do speaker diarization; transcripts are a single stream.
- Output is plain text only. SRT/VTT subtitles and JSON output are not implemented.
- One file at a time, single-threaded — running multiple Whisper processes can
  exhaust RAM/VRAM and slow the machine down.

## License notes

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — MIT
- Whisper models — MIT (see the Whisper model card)
- ffmpeg — LGPL/GPL (system package)
