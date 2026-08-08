#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
}


def find_media_files(input_dir):
    files = []
    for root, _dirs, names in os.walk(input_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files, key=str.lower)


def extract_audio(ffmpeg, media_file, wav_file):
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            media_file,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            wav_file,
        ],
        check=True,
        capture_output=True,
    )


def media_duration(ffprobe, media_file):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            media_file,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def format_duration(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def available_memory_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def wait_for_memory(per_worker_mb):
    while True:
        avail = available_memory_mb()
        if avail is None or avail >= per_worker_mb:
            return
        time.sleep(5)


def transcribe(whisper_cli, model, wav_file, language, out_prefix):
    subprocess.run(
        [
            whisper_cli,
            "-m",
            model,
            "-f",
            wav_file,
            "-l",
            language,
            "-otxt",
            "-of",
            out_prefix,
            "-nt",
        ],
        check=True,
        capture_output=True,
    )
    with open(out_prefix + ".txt", encoding="utf-8") as f:
        return f.read()


def process_file(
    media_file,
    i,
    total,
    args,
    input_dir,
    output_dir,
    whisper_cli,
    ffmpeg,
    ffprobe,
    per_worker_mb,
    lock,
    state,
    started,
):
    rel_dir = os.path.relpath(os.path.dirname(media_file), input_dir)
    base = os.path.splitext(os.path.basename(media_file))[0]
    out_file = os.path.join(output_dir, rel_dir, base + ".txt")

    if (
        os.path.exists(out_file)
        and os.path.getsize(out_file) > 0
        and not args.overwrite
    ):
        with lock:
            print(f"[{i}/{total}] SKIP {os.path.relpath(media_file, input_dir)}")
            state["skipped"] += 1
        return

    with lock:
        print(f"[{i}/{total}] {os.path.relpath(media_file, input_dir)}")

    if ffprobe:
        duration = media_duration(ffprobe, media_file)
        if duration is not None:
            with lock:
                print(f"    Duration: {format_duration(duration)}")

    wait_for_memory(per_worker_mb)

    tmp = tempfile.mkdtemp(prefix="transcribe-")
    wav_file = os.path.join(tmp, "audio.wav")
    try:
        with lock:
            print("    Extracting audio...")
        extract_audio(ffmpeg, media_file, wav_file)

        with lock:
            print("    Transcribing Arabic...")
        out_prefix = os.path.join(tmp, "out")
        transcript = transcribe(
            whisper_cli, args.model, wav_file, args.language, out_prefix
        )

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(transcript.strip() + "\n")
        with lock:
            print(f"    Saved: {out_file}")
            state["processed"] += 1
            processed = state["processed"]
            if processed >= 1:
                remaining = total - i
                eta = time.time() + (time.time() - started) / processed * remaining
                print(f"    ETA: {time.strftime('%H:%M:%S', time.localtime(eta))}")
    except subprocess.CalledProcessError as e:
        with lock:
            print(f"    ERROR: {e.stderr.decode(errors='replace').strip()}")
            state["failed"] += 1
            state["failed_files"].append(media_file)
    except Exception as e:
        with lock:
            print(f"    ERROR: {e}")
            state["failed"] += 1
            state["failed_files"].append(media_file)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Local Arabic speech-to-text via whisper.cpp"
    )
    parser.add_argument(
        "-input",
        "--input",
        default="input",
        help="source media directory (default: input)",
    )
    parser.add_argument(
        "--output", default="text", help="transcript directory (default: text)"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(SCRIPT_DIR, "whisper.cpp", "models", "ggml-small.bin"),
        help="whisper.cpp GGML model path",
    )
    parser.add_argument(
        "--language", default="ar", help="whisper language code (default: ar)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-transcribe files that already have a transcript",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel worker count (default: CPU cores)",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        sys.exit("--jobs must be >= 1")

    input_dir = args.input
    output_dir = args.output
    whisper_cli = os.path.join(SCRIPT_DIR, "whisper.cpp", "build", "bin", "whisper-cli")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if ffmpeg is None:
        sys.exit("ffmpeg not found on PATH")
    if not os.path.exists(whisper_cli):
        sys.exit(f"whisper-cli not found: {whisper_cli}")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")
    if not os.path.isdir(input_dir):
        sys.exit(f"input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    media_files = find_media_files(input_dir)
    print(f"Scanning {input_dir}/...")
    print(f"Found {len(media_files)} media files.\n")

    started = time.time()
    print(f"Started at {time.strftime('%H:%M:%S')}")

    state = {"processed": 0, "skipped": 0, "failed": 0, "failed_files": []}
    lock = threading.Lock()
    per_worker_mb = max(os.path.getsize(args.model) / (1024 * 1024) * 2, 512)
    print(
        f"Using {args.jobs} worker(s), ~{int(per_worker_mb)} MB reserved per worker.\n"
    )

    kwargs = dict(
        args=args,
        input_dir=input_dir,
        output_dir=output_dir,
        whisper_cli=whisper_cli,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        per_worker_mb=per_worker_mb,
        lock=lock,
        state=state,
        started=started,
    )
    total = len(media_files)

    if args.jobs <= 1:
        for i, media_file in enumerate(media_files, 1):
            process_file(media_file, i, total, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(process_file, media_file, i, total, **kwargs)
                for i, media_file in enumerate(media_files, 1)
            ]
            for future in futures:
                future.result()

    processed = state["processed"]
    skipped = state["skipped"]
    failed = state["failed"]
    failed_files = state["failed_files"]

    print("\nFinished.")
    elapsed = time.time() - started
    print(
        f"Finished at {time.strftime('%H:%M:%S')} (took {int(elapsed // 60)}m {int(elapsed % 60)}s)"
    )
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    for path in failed_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
