"""Resolve the frozen LoRA baseline and the isolated visual-attention treatment."""
from __future__ import annotations

import re

BASELINE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_ATTENTION = re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.attn\.(qkv|proj)$")
# The vision MLP. Qwen2.5-VL names it gate/up/down_proj, so the baseline suffixes already
# adapt it; Qwen2-VL (fc1/fc2) and Qwen3-VL (linear_fc1/2) use other names, so the baseline
# leaves their vision encoder frozen. "vision-mlp" closes that gap and is a no-op on Qwen2.5.
VISION_MLP = re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.mlp\.[a-z_0-9]+$")
# Kimi-VL's MoE decoder reuses gate/up/down_proj for 64 routed experts per layer, so the
# baseline suffixes would adapt ~5k NF4 expert Linears. "kimi" adapts the MoonViT blocks,
# attention, and the dense/shared-expert MLPs, and leaves the routed experts frozen.
KIMI = re.compile(r"(?:^|\.)(vision_tower\.encoder\.blocks|language_model\.model\.layers)\.(\d+)\."
                  r"(wqkv|wo|mlp\.fc[01]|self_attn\.(?:q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"
                  r"|mlp\.(?:shared_experts\.)?(?:gate|up|down)_proj)$")
KIMI_BLOCK = re.compile(r"(?:^|\.)(vision_tower\.encoder\.blocks|language_model\.model\.layers)\.(\d+)$")


def resolve_targets(model, scope="baseline", rank=16):
    import torch

    if scope not in {"baseline", "vision-attention", "vision-mlp", "kimi"} or rank <= 0:
        raise ValueError("Unknown LoRA scope or invalid rank")
    modules = dict(model.named_modules())
    baseline = [name for name in modules if name.rsplit(".", 1)[-1] in BASELINE_TARGETS]
    if not baseline:
        raise ValueError("No baseline LoRA targets found")
    added = []
    if scope == "kimi":
        baseline = []
        added = [name for name in modules if KIMI.search(name)]
        blocks = [KIMI_BLOCK.search(name).groups() for name in modules if KIMI_BLOCK.search(name)]
        if not blocks:
            raise ValueError("No Kimi-VL vision blocks or decoder layers found")
        # Four vision Linears per MoonViT block, four MLA projections per decoder layer.
        for tower, index in blocks:
            kinds = {KIMI.search(n)[3] for n in added if KIMI.search(n).group(1, 2) == (tower, index)}
            expected = ({"wqkv", "wo", "mlp.fc0", "mlp.fc1"} if tower.startswith("vision") else
                        {f"self_attn.{k}" for k in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")})
            if not expected <= kinds:
                raise ValueError(f"Kimi-VL {tower}.{index} lacks adapters for {sorted(expected - kinds)}")
    if scope == "vision-attention":
        added = [name for name in modules if VISION_ATTENTION.search(name)]
        indices = {(int(VISION_ATTENTION.search(name)[1]), name.rsplit(".", 1)[-1]) for name in added}
        if len(added) != 64 or indices != {(i, kind) for i in range(32) for kind in ("qkv", "proj")}:
            raise ValueError("Expected exactly 64 visual attention modules in 32 blocks")
    if scope == "vision-mlp":
        mlp = [name for name in modules
               if VISION_MLP.search(name) and isinstance(modules[name], torch.nn.Linear)]
        if not mlp:
            raise ValueError("No vision MLP Linear layers found")
        added = [name for name in mlp if name not in baseline]
        uncovered = ({int(VISION_MLP.search(n)[1]) for n in mlp}
                     - {int(VISION_MLP.search(n)[1]) for n in mlp if n in baseline or n in added})
        if uncovered:
            raise ValueError(f"Vision blocks without an adapted MLP layer: {sorted(uncovered)}")
    for name in baseline + added:
        if not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"LoRA target is not Linear: {name}")
    for name in (added if scope == "vision-attention" else []):
        module = modules[name]
        expected_out = 3840 if name.endswith(".qkv") else 1280
        if (module.in_features, module.out_features) != (1280, expected_out):
            raise ValueError(f"Unexpected pinned 3B attention dimensions: {name}")
    count = lambda names: sum(rank * (modules[n].in_features + modules[n].out_features) for n in names)
    audit = dict(scope=scope, rank=rank, baseline_modules=baseline, added_modules=added,
                 resolved_modules=baseline + added, added_module_count=len(added),
                 baseline_adapter_parameters=count(baseline), added_adapter_parameters=count(added),
                 expected_adapter_parameters=count(baseline + added))
    if scope == "vision-attention" and count(added) != 245760 * rank:
        raise ValueError("Unexpected visual-attention adapter parameter count")
    # Keep the original suffix list (including its order) for the default path.
    targets = list(BASELINE_TARGETS) if scope == "baseline" else baseline + added
    return targets, audit


class AttentionGradientAudit:
    """Observe B gradients: A legitimately has zero gradient at zero-B initialization.

    Reductions remain on device until report(), avoiding 64 GPU synchronizations
    per backward pass. This observes gradients without modifying them.
    """
    def __init__(self, model, added_modules):
        import torch

        self.stats, self.handles = {}, []
        for name in added_modules:
            matches = [(n, p) for n, p in model.named_parameters()
                       if n.endswith(name + ".lora_B.default.weight") and p.requires_grad]
            if len(matches) != 1:
                raise ValueError(f"Expected one trainable B adapter for {name}")
            parameter_name, parameter = matches[0]
            state = {"parameter": parameter_name, "calls": 0, "finite": None, "max_abs": None}
            self.stats[name] = state

            def observe(gradient, state=state):
                value = gradient.detach()
                finite, maximum = torch.isfinite(value).all(), value.abs().amax()
                state["finite"] = finite if state["finite"] is None else state["finite"] & finite
                state["max_abs"] = maximum if state["max_abs"] is None else torch.maximum(state["max_abs"], maximum)
                state["calls"] += 1

            self.handles.append(parameter.register_hook(observe))

    def report(self):
        modules = {name: dict(parameter=s["parameter"], backward_calls=s["calls"],
                             finite=bool(s["finite"].item()) if s["finite"] is not None else False,
                             max_abs=float(s["max_abs"].item()) if s["max_abs"] is not None else 0.0)
                   for name, s in self.stats.items()}
        return {"passed": bool(modules) and all(s["finite"] and s["max_abs"] > 0 for s in modules.values()),
                "observed": "lora_B.default.weight; pre-unscale gradients", "modules": modules}

    def close(self):
        for handle in self.handles:
            handle.remove()
