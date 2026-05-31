from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(self, vision_hidden_size, text_hidden_size, num_image_tokens):
        super().__init__()
        self.num_image_tokens = num_image_tokens
        self.text_hidden_size = text_hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, num_image_tokens * text_hidden_size),
        )

    def forward(self, vision_hidden_states):
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        out = self.proj(vision_hidden_states.mean(dim=1))
        b = vision_hidden_states.shape[0]
        return out.view(b, self.num_image_tokens, self.text_hidden_size)


def merge_visual_embeddings(input_embeds, input_ids, visual_embeds, image_token_id):
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    merged = input_embeds.clone()
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        for k, pos in enumerate(positions):
            merged[b, pos] = visual_embeds[b, k]
    return merged


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(32 * 4 * 4, hidden_size)

    def forward(self, pixel_values):
        b, t, c, h, w = pixel_values.shape
        x = self.conv(pixel_values.view(b * t, c, h, w)).flatten(1)
        return SimpleNamespace(last_hidden_state=self.proj(x).view(b, t, self.hidden_size))


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        logits = self.lm_head(self.layers(inputs_embeds))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder, language_model, config):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self):
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_batch(self, batch):
        pixel_values = batch["pixel_values"]
        if pixel_values.ndim == 5:
            b, t, c, h, w = pixel_values.shape
            if hasattr(self.vision_encoder, "config"):
                pv = pixel_values.view(b * t, c, h, w)
                hidden = self.vision_encoder(pixel_values=pv).last_hidden_state
                hidden = hidden.view(b, t * hidden.shape[1], hidden.shape[2])
            else:
                hidden = self.vision_encoder(pixel_values).last_hidden_state
        elif hasattr(self.vision_encoder, "config"):
            hidden = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        else:
            hidden = self.vision_encoder(pixel_values).last_hidden_state

        visual_embeds = self.adapter(hidden)
        input_ids = batch["input_ids"]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged = merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)
        return merged, batch["attention_mask"], batch.get("labels")

    def forward(self, batch):
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        inputs_embeds, attention_mask, labels = self._encode_batch(batch)
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, batch, **generation_kwargs):
        """Generate answer token ids."""
        max_new_tokens = int(generation_kwargs.get("max_new_tokens", 16))
        inputs_embeds, attention_mask, _ = self._encode_batch(batch)
        generated = batch["input_ids"]
        cur_embeds = inputs_embeds
        cur_mask = attention_mask

        for _ in range(max_new_tokens):
            next_id = self.language_model(inputs_embeds=cur_embeds, attention_mask=cur_mask).logits[:, -1].argmax(
                dim=-1, keepdim=True
            )
            generated = torch.cat([generated, next_id], dim=1)
            cur_embeds = torch.cat([cur_embeds, self.language_model.get_input_embeddings()(next_id)], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(next_id)], dim=1)

        return generated
