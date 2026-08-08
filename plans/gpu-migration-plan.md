# GPU Migration Plan

Goal: let `transcribe.py` use the NVIDIA GPU for whisper inference instead of CPU-only execution.

## Current State

- `transcribe.py` runs `whisper.cpp/build/bin/whisper-cli` (CPU-only build, `GGML_CUDA=OFF`).
- GPU never used: `nvidia-smi` reports 0 MiB VRAM in use during runs.
- Why: GPU backends in whisper.cpp are compiled in at build time. The current binary links
  only `libggml-cpu.so` — no CUDA/Vulkan/OpenCL backend exists to enable at runtime.

## Hardware / Environment Audit

| Component | State |
|---|---|
| GPU | NVIDIA GeForce 940MX, 2 GB VRAM, compute capability 5.0 (Maxwell) |
| Driver | 550.163.01 (supports CUDA 12.x; CUDA 13 dropped Maxwell) |
| CUDA toolkit (`nvcc`) | Not installed |
| CUDA runtime (`libcudart`/`libcublas`) | Not on disk anywhere |
| Vulkan loader | Present, but shader compiler (`glslc`/`shaderc`) absent |
| Toolchain | gcc/g++/cmake present |

Zero-install conclusion: **not possible.** Any GPU backend must be compiled in, and that
requires tooling this box lacks (`nvcc` for CUDA, `shaderc` for Vulkan). A downloaded
prebuilt CUDA binary would still fail at runtime without `libcudart`/`libcublas`.

## Plan

### 1. Install CUDA toolkit (one-time, requires sudo + network)

```sh
sudo apt install nvidia-cuda-toolkit   # 12.4, Maxwell (sm_50) compatible
```

### 2. Rebuild whisper.cpp with CUDA — separate dir, keep CPU build intact

```sh
cmake -B whisper.cpp/build-cuda -DGGML_CUDA=ON
cmake --build whisper.cpp/build-cuda -j$(nproc)
```

Produces `whisper.cpp/build-cuda/bin/whisper-cli` with the CUDA backend.

### 3. `transcribe.py` changes

- Prefer `whisper.cpp/build-cuda/bin/whisper-cli` when it exists; fall back to the CPU
  build automatically if it doesn't.
- When the GPU build is in use, pass `-ngl 99` to whisper-cli (offload all layers;
  `ggml-small.bin` fits in 2 GB VRAM). CPU path unchanged.
- Extend the adaptive throttle (`wait_for_memory`) to also require free GPU VRAM
  (`nvidia-smi --query-gpu=memory.free`) >= model size before a worker starts.
  If `nvidia-smi` is unavailable, degrade to today's system-RAM behavior.

## Known Trade-offs

- On this 940MX, GPU inference may be *slower* than CPU (Maxwell-class laptop GPU,
  no fast FP16, ~0.6 TFLOPS FP32).
- 2 GB VRAM fits exactly one model, so parallel `--jobs` GPU workers will serialize on
  VRAM (the throttle enforces this instead of crashing). CPU-only jobs stay parallel.

## Deferred (ponytail)

- `--gpu-layers` knob (all-layers hardcoded; add when a bigger model arrives).
- Dynamic per-layer VRAM sizing.
- GPU-supporting prebuilt binary distribution (needs libcudart/cublas at runtime).

## Acceptance

- `nvidia-smi` shows non-zero VRAM/GPU utilization while `transcribe.py` runs with the
  GPU build.
- Existing CPU build still works as a fallback.
