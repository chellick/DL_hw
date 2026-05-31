# Report

## Track

Выбранный трек:

```text
A + B (MPS)
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py
- [x] Track B: ViT + Qwen2-0.5B, adapter-only, safetensors checkpoint

## Конфигурация

Track A (smoke):

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: local=1, global=1
```

Track B (MPS):

```text
config path: configs/track_b_mps.yaml
seed: 42
device: mps (auto)
dtype: float32
max_steps: 300 (smoke: 10)
batch size: local=1, global=8
vision: google/vit-base-patch16-224
LLM: Qwen/Qwen2-0.5B-Instruct
checkpoint: artifacts/adapter.safetensors
```

## Результаты

```text
public tests: 14 passed (pytest -q tests_public)
Track A train: finite loss, --fast-train OK
Track B train: finite loss on MPS, adapter saved
Track B benchmark (toy-dev, 10 steps): overall 1.0
```

## Использованные ресурсы

```text
CPU/GPU: MPS
VRAM/unified memory: 2-3 GB (ViT-base + Qwen2-0.5B + adapter)
время обучения: 2 мин (10 steps), полный run 300 steps 30-60 мин
```

## Анализ ошибок

1. После короткого обучения (10 steps) модель переобучается на toy-dev, на medium/mathvista качество будет ниже.
2. ViT видит только один тайл 224×224, мелкие детали на схемах могут теряться.
3. Adapter-only без LoRA ограничивает выразительность: LLM frozen, ошибки на сложных visual reasoning задачах ожидаемы.

## Комментарии

Track A: mock-модели для CPU smoke-check. Track B: реальные HF-модели через `device: auto` (MPS). Для полного Track B можно запустить `python -m hw.train --config configs/track_b_mps.yaml` или medium-вариант `configs/track_b_mps_medium.yaml`, затем `python -m hw.benchmark --config configs/inference_mps.yaml`.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
