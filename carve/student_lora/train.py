from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .losses import bradley_terry_loss, masked_huber_loss


@dataclass
class ScoreBatch:
    model_inputs: dict[str, Tensor]
    targets: Tensor
    target_mask: Tensor

    def to(self, device: torch.device | str) -> "ScoreBatch":
        return ScoreBatch(
            model_inputs={key: value.to(device) for key, value in self.model_inputs.items()},
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
        )


@dataclass
class RankingBatch:
    model_inputs: dict[str, Tensor]
    preferred_indices: Tensor
    rejected_indices: Tensor

    def to(self, device: torch.device | str) -> "RankingBatch":
        return RankingBatch(
            model_inputs={key: value.to(device) for key, value in self.model_inputs.items()},
            preferred_indices=self.preferred_indices.to(device),
            rejected_indices=self.rejected_indices.to(device),
        )


@dataclass
class TrainStepResult:
    total_loss: Tensor
    regression_loss: Tensor
    ranking_loss: Tensor
    masked_targets: int
    ranking_pairs: int


def score_batch(model: nn.Module, batch: ScoreBatch | RankingBatch, *, event_micro_batch_size: int = 2) -> Tensor:
    inputs = batch.model_inputs
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if input_ids is None or attention_mask is None or input_ids.shape[0] <= event_micro_batch_size:
        return model.score_graph(**inputs)
    if not hasattr(model, "encode_events") or not hasattr(model, "graph_head"):
        return model.score_graph(**inputs)
    text_states = []
    for start in range(0, input_ids.shape[0], event_micro_batch_size):
        stop = min(start + event_micro_batch_size, input_ids.shape[0])
        text_states.append(model.encode_events(input_ids[start:stop], attention_mask[start:stop]))
    joined = torch.cat(text_states, dim=0)
    return model.graph_head(
        joined,
        inputs["event_type_ids"],
        inputs["role_ids"],
        inputs["numeric_features"],
        inputs["edge_index"],
        inputs["edge_types"],
    )


def train_step(
    model: nn.Module,
    factual_batch: ScoreBatch,
    ranking_batch: RankingBatch | None,
    *,
    beta: float = 0.2,
    huber_delta: float = 1.0,
    event_micro_batch_size: int = 2,
) -> TrainStepResult:
    predictions = score_batch(model, factual_batch, event_micro_batch_size=event_micro_batch_size)
    targets = factual_batch.targets.to(predictions.device)
    mask = factual_batch.target_mask.to(predictions.device) & torch.isfinite(targets)
    regression_loss = masked_huber_loss(predictions, targets, mask, delta=huber_delta)
    ranking_pairs = 0
    if ranking_batch is None or ranking_batch.preferred_indices.numel() == 0:
        ranking_loss = (predictions.sum() * 0.0)
    else:
        ranking_scores = score_batch(model, ranking_batch, event_micro_batch_size=event_micro_batch_size)
        preferred_indices = ranking_batch.preferred_indices.to(ranking_scores.device)
        rejected_indices = ranking_batch.rejected_indices.to(ranking_scores.device)
        preferred = ranking_scores[preferred_indices]
        rejected = ranking_scores[rejected_indices]
        ranking_pairs = int(preferred.numel())
        ranking_loss = bradley_terry_loss(preferred, rejected)
    total_loss = regression_loss + beta * ranking_loss
    return TrainStepResult(total_loss, regression_loss, ranking_loss, int(mask.sum().item()), ranking_pairs)


def make_optimizer(model: nn.Module, *, lora_lr: float = 2e-4, head_lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    lora_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name.lower():
            lora_parameters.append(parameter)
        elif name.startswith("graph_head.") or ".graph_head." in name:
            head_parameters.append(parameter)
        else:
            raise ValueError(f"unexpected trainable parameter: {name}")
    if not lora_parameters or not head_parameters:
        raise ValueError("optimizer requires non-empty LoRA and graph-head parameter groups")
    return torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": lora_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def save_student_checkpoint(model: nn.Module, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "student_adapter"
    if hasattr(model, "backbone") and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(adapter_dir)
    graph_path = output_dir / "graph_head.pt"
    graph_head = getattr(model, "graph_head", None)
    if graph_head is None:
        raise ValueError("model has no graph_head")
    torch.save(graph_head.state_dict(), graph_path)
    return {"adapter": str(adapter_dir), "graph_head": str(graph_path)}
