import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig, TinyLanguageModel, TinyVisionEncoder
from hw.processor import MathVLMProcessor, ProcessorConfig

VOCAB_SIZE = 4096
SPECIAL_TOKENS = [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(name, device):
    if name == "float16" and device.type == "cpu":
        return torch.float32
    return getattr(torch, name)


def is_mock_config(config):
    return config["model"].get("vision_encoder") == "tiny/local-or-mocked"


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<image>": 2, "<image_start>": 3, "<image_end>": 4}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for token in text.replace("\n", " ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids if i not in {self.pad_token_id, self.eos_token_id})

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def build_tokenizer(config):
    if is_mock_config(config):
        return SimpleTokenizer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["language_model"], trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_processor(config, tokenizer):
    proc_cfg = dict(config["processor"])
    if not is_mock_config(config) and "vit_normalize" not in proc_cfg:
        proc_cfg["vit_normalize"] = True
    return MathVLMProcessor(tokenizer, ProcessorConfig(**proc_cfg))


def build_model(config, tokenizer):
    if is_mock_config(config):
        hidden = 64
        vision_encoder = TinyVisionEncoder(hidden_size=hidden)
        language_model = TinyLanguageModel(vocab_size=VOCAB_SIZE, hidden_size=hidden)
        model_config = ModelConfig(
            vision_hidden_size=hidden,
            text_hidden_size=hidden,
            num_image_tokens=config["processor"]["num_image_tokens"],
            image_token_id=tokenizer.vocab["<image>"],
        )
        return MathVLM(vision_encoder, language_model, model_config)

    from transformers import AutoModel, AutoModelForCausalLM

    model_cfg = config["model"]
    vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"])
    language_model = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"], trust_remote_code=True)
    language_model.resize_token_embeddings(len(tokenizer))
    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=config["processor"]["num_image_tokens"],
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )
    return MathVLM(vision_encoder, language_model, model_config)


def load_adapter(model, path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        model.adapter.load_state_dict(load_file(str(path)))
    else:
        model.adapter.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def save_adapter(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(model.adapter.state_dict(), str(path))
    else:
        torch.save(model.adapter.state_dict(), path)


def train_one_step(model, batch, optimizer):
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    out = model(batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.detach())


def run_training(config, fast_train=False):
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    data_cfg = config["data"]
    trainer_cfg = config["trainer"]
    device = resolve_device(trainer_cfg.get("device", "cpu"))
    dtype = resolve_dtype(trainer_cfg.get("dtype", "float32"), device)

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )
    tokenizer = build_tokenizer(config)
    processor = build_processor(config, tokenizer)
    model = build_model(config, tokenizer).to(device=device, dtype=dtype)
    model.freeze_backbones()

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
    )
    loader = DataLoader(
        dataset,
        batch_size=trainer_cfg["local_batch_size"],
        shuffle=True,
        num_workers=trainer_cfg.get("num_workers", 0),
        collate_fn=lambda batch: processor.collate([processor(sample) for sample in batch]),
    )

    max_steps = trainer_cfg.get("max_steps", 3)
    if fast_train:
        max_steps = min(max_steps, 3)

    accum_steps = max(1, trainer_cfg.get("global_batch_size", 1) // trainer_cfg.get("local_batch_size", 1))
    step = 0
    optimizer.zero_grad()

    for batch in loader:
        if step >= max_steps:
            break
        batch = {k: v.to(device=device, dtype=dtype if k == "pixel_values" else None) for k, v in batch.items()}
        model.train()
        out = model(batch)
        loss = out.loss / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        step += 1

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        save_adapter(model, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
