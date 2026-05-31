from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX
    vit_normalize: bool = False


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image):
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB").resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.BILINEAR,
        )
        arr = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        if self.config.vit_normalize:
            arr = (arr - 0.5) / 0.5
        return arr.unsqueeze(0)

    def build_prompt(self, sample, include_answer):
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        prompt = (
            f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}\n"
            f"{sample.question}\n"
            f"{'\n'.join(sample.options)}\n"
            "Ответ:"
        )
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample):
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        full_prompt = self.build_prompt(sample, include_answer=True)
        prefix_prompt = self.build_prompt(sample, include_answer=False)
        encoded = self.tokenizer(
            full_prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_length,
        )
        prefix_len = len(
            self.tokenizer(prefix_prompt, add_special_tokens=False, truncation=True, max_length=self.config.max_length)[
                "input_ids"
            ]
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        labels[:prefix_len] = self.config.ignore_index
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": labels,
        }

    def __call__(self, sample):
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch):
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        pad_id = self.tokenizer.pad_token_id
        ignore_index = self.config.ignore_index
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_len,), ignore_index, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        }
