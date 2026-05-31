import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from hw.constants import CHOICES, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.processor import MathVLMProcessor, ProcessorConfig
from hw.train import (
    SimpleTokenizer,
    build_model,
    build_processor,
    build_tokenizer,
    load_adapter,
    resolve_device,
)


def parse_mc_answer(text, choices=CHOICES):
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    pattern = r"\b([" + "".join(choices) + r"])\b|\([" + "".join(choices) + r"]\)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    if match.group(1):
        return match.group(1).upper()
    return match.group(0)[1].upper()


def build_benchmark_prompt(question, options):
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def _benchmark_text(sample, num_image_tokens):
    image_tokens = " ".join([IMAGE_TOKEN] * num_image_tokens)
    prompt = build_benchmark_prompt(sample.question, sample.options)
    return f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n{prompt}"


def compute_accuracy(rows):
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config, toy=False):
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})
    device = resolve_device("cpu" if toy else inference_cfg.get("device", "auto"))

    dataset = MathVQADataset(
        data_cfg["eval_manifest"],
        split=data_cfg.get("split", "dev"),
        max_samples=data_cfg.get("max_samples"),
    )
    processor_cfg = ProcessorConfig(**config["processor"])
    if toy:
        tokenizer = SimpleTokenizer()
        processor = MathVLMProcessor(tokenizer, processor_cfg)
    else:
        tokenizer = build_tokenizer(config)
        processor = build_processor(config, tokenizer)
    model = build_model(config, tokenizer).to(device)
    adapter_path = config.get("model", {}).get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        load_adapter(model, adapter_path)
    model.eval()

    rows = []
    max_new_tokens = int(inference_cfg.get("max_new_tokens", 64))

    for sample in dataset:
        encoded = tokenizer(
            _benchmark_text(sample, processor_cfg.num_image_tokens),
            add_special_tokens=False,
            truncation=True,
            max_length=processor_cfg.max_length,
        )
        batch = {
            "input_ids": torch.tensor([encoded["input_ids"]], device=device),
            "attention_mask": torch.tensor([encoded["attention_mask"]], device=device),
            "pixel_values": processor.preprocess_image(sample.image).unsqueeze(0).to(device),
        }

        generated = model.generate(batch, max_new_tokens=max_new_tokens)
        new_ids = generated[0, batch["input_ids"].shape[1] :].tolist()
        if hasattr(tokenizer, "decode") and not isinstance(tokenizer, SimpleTokenizer):
            output_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        else:
            output_text = tokenizer.decode(new_ids)
        rows.append(
            {
                "id": sample.id,
                "answer": sample.answer,
                "prediction": parse_mc_answer(output_text),
                "subject": sample.subject,
                "output": output_text,
            }
        )

    output_path = inference_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
