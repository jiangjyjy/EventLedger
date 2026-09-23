from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def validate_trainable_parameter_names(parameters: list[tuple[str, bool]]) -> None:
    for name, trainable in parameters:
        lowered = name.lower()
        if trainable and not ("lora_" in lowered or lowered.startswith("graph_head.") or ".graph_head." in lowered):
            raise ValueError(f"unexpected trainable base parameter: {name}")


def place_model_on_device(model: nn.Module, device: torch.device | str) -> nn.Module:
    model.to(device)
    return model


class RelationalGraphRewardHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_event_types: int,
        num_roles: int,
        num_relations: int,
        numeric_dim: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.event_type_embedding = nn.Embedding(num_event_types, hidden_dim)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        self.relation_src = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.relation_value = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.relation_query = nn.Parameter(torch.empty(num_relations, hidden_dim, hidden_dim))
        self.attention_vector = nn.Parameter(torch.empty(num_relations, hidden_dim * 2))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reward_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_src)
        nn.init.xavier_uniform_(self.relation_value)
        nn.init.xavier_uniform_(self.relation_query)
        nn.init.xavier_uniform_(self.attention_vector)

    def encode_nodes(
        self,
        text_states: Tensor,
        event_type_ids: Tensor,
        role_ids: Tensor,
        numeric_features: Tensor,
    ) -> Tensor:
        dtype = self.input_proj.weight.dtype
        device = text_states.device
        text_states = text_states.to(device=device, dtype=dtype)
        event_type_ids = event_type_ids.to(device=device, dtype=torch.long)
        role_ids = role_ids.to(device=device, dtype=torch.long)
        numeric_features = numeric_features.to(device=device, dtype=dtype)
        return (
            self.input_proj(text_states)
            + self.event_type_embedding(event_type_ids)
            + self.role_embedding(role_ids)
            + self.numeric_proj(numeric_features)
        )

    def propagate(self, node_states: Tensor, edge_index: Tensor, edge_types: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return self.norm(node_states)
        messages: list[Tensor] = []
        scores: list[Tensor] = []
        destinations: list[int] = []
        for edge_index_value, relation in zip(edge_index.T, edge_types, strict=True):
            source = int(edge_index_value[0])
            destination = int(edge_index_value[1])
            relation_index = int(relation)
            source_state = node_states[source] @ self.relation_src[relation_index]
            value = node_states[source] @ self.relation_value[relation_index]
            query = node_states[destination] @ self.relation_query[relation_index]
            attention_input = torch.cat((source_state, query), dim=-1)
            score = F.leaky_relu((attention_input * self.attention_vector[relation_index]).sum(), negative_slope=0.2)
            messages.append(value)
            scores.append(score)
            destinations.append(destination)

        aggregate = torch.zeros_like(node_states)
        for destination in sorted(set(destinations)):
            indices = [i for i, value in enumerate(destinations) if value == destination]
            weights = torch.softmax(torch.stack([scores[i] for i in indices]), dim=0)
            for weight, index in zip(weights, indices, strict=True):
                aggregate[destination] = aggregate[destination] + weight * messages[index]
        return self.norm(node_states + self.dropout(F.gelu(aggregate)))

    def forward(
        self,
        text_states: Tensor,
        event_type_ids: Tensor,
        role_ids: Tensor,
        numeric_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
    ) -> Tensor:
        node_states = self.encode_nodes(text_states, event_type_ids, role_ids, numeric_features)
        graph_states = self.propagate(node_states, edge_index, edge_types)
        return self.reward_head(graph_states).squeeze(-1)


class QwenLoRAGraphPRM(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer: object, *, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        config = backbone.config
        text_config = getattr(config, "text_config", config)
        self.qwen_hidden_size = int(getattr(text_config, "hidden_size", getattr(config, "hidden_size")))
        self.graph_head = RelationalGraphRewardHead(
            input_dim=self.qwen_hidden_size,
            hidden_dim=hidden_dim,
            num_event_types=10,
            num_roles=10,
            num_relations=7,
        )

    @classmethod
    def from_local(
        cls,
        model_path: str,
        *,
        hidden_dim: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: torch.device | str | None = None,
    ) -> "QwenLoRAGraphPRM":
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        backbone.config.use_cache = False
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        linear_suffixes = {
            name.rsplit(".", 1)[-1]
            for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear)
        }
        preferred_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        targets = [name for name in preferred_targets if name in linear_suffixes]
        if not targets:
            raise RuntimeError("Qwen3.5 has no supported linear LoRA target modules")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        backbone = get_peft_model(backbone, peft_config)
        model = cls(backbone, tokenizer, hidden_dim=hidden_dim)
        if device is not None:
            place_model_on_device(model, device)
        validate_trainable_parameter_names([(name, parameter.requires_grad) for name, parameter in model.named_parameters()])
        return model

    def trainable_summary(self) -> dict[str, int | str | list[str]]:
        trainable = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for _, parameter in trainable)
        base_trainable = sum(
            parameter.numel()
            for name, parameter in trainable
            if "lora_" not in name.lower() and not (name.startswith("graph_head.") or ".graph_head." in name)
        )
        return {
            "dtype": "bfloat16",
            "backbone": "Qwen/Qwen3.5-9B-Base",
            "trainable_parameters": trainable_count,
            "ordinary_base_trainable_parameters": base_trainable,
            "trainable_parameter_names": [name for name, _ in trainable],
        }

    def encode_events(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        last_indices = attention_mask.sum(dim=1).clamp_min(1).to(torch.long) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_indices, last_indices]

    def score_graph(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        event_type_ids: Tensor,
        role_ids: Tensor,
        numeric_features: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
    ) -> Tensor:
        text_states = self.encode_events(input_ids, attention_mask)
        return self.graph_head(text_states, event_type_ids, role_ids, numeric_features, edge_index, edge_types)
