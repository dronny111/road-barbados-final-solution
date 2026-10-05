"""Exercise actual PEFT selection and gradient flow without downloading weights."""
import importlib.util
import unittest

from road_ocr.lora import BASELINE_TARGETS, AttentionGradientAudit, resolve_targets


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch optional")
class LoraScopeTest(unittest.TestCase):
    @staticmethod
    def model(mlp_names=("gate_proj", "up_proj", "down_proj")):
        import torch
        nn = torch.nn
        # Meta tensors give us the real pinned dimensions with no allocation.
        with torch.device("meta"):
            model = nn.Module()
            model.visual = nn.Module()
            model.visual.patch_embed = nn.Module()
            model.visual.patch_embed.proj = nn.Conv3d(3, 1280, 2)
            model.visual.blocks = nn.ModuleList()
            for _ in range(32):
                block = nn.Module()
                block.attn = nn.Module()
                block.attn.qkv = nn.Linear(1280, 3840)
                block.attn.proj = nn.Linear(1280, 1280)
                block.mlp = nn.Module()
                for name in mlp_names:
                    setattr(block.mlp, name, nn.Linear(1280, 3420))
                block.mlp.act_fn = nn.GELU()  # not Linear: must never be targeted
                model.visual.blocks.append(block)
            model.language = nn.Module()
            for name in BASELINE_TARGETS:
                setattr(model.language, name, nn.Linear(16, 16))
        return model

    def test_default_preserves_suffixes_and_treatment_adds_exact_linear_modules(self):
        model = self.model()
        targets, baseline = resolve_targets(model)
        self.assertEqual(targets, list(BASELINE_TARGETS))
        explicit, treatment = resolve_targets(model, "vision-attention")
        self.assertEqual(treatment["baseline_modules"], baseline["resolved_modules"])
        self.assertEqual(len(set(explicit) - set(baseline["resolved_modules"])), 64)
        self.assertEqual(treatment["added_adapter_parameters"], 3932160)
        self.assertNotIn("visual.patch_embed.proj", explicit)
        self.assertTrue(all(n.startswith("visual.blocks.") for n in treatment["added_modules"]))

    def test_vision_mlp_adapts_every_family_and_is_a_no_op_on_qwen25(self):
        for names, expected_added in ((("gate_proj", "up_proj", "down_proj"), 0),  # Qwen2.5-VL
                                      (("fc1", "fc2"), 64),                        # Qwen2-VL
                                      (("linear_fc1", "linear_fc2"), 64)):         # Qwen3-VL
            _, audit = resolve_targets(self.model(names), "vision-mlp")
            self.assertEqual(audit["added_module_count"], expected_added, names)
            self.assertFalse(any(n.endswith("act_fn") for n in audit["resolved_modules"]))
            covered = {n.split(".")[2] for n in audit["resolved_modules"] if n.startswith("visual.blocks.")}
            self.assertEqual(len(covered), 32, names)  # every vision block has an adapted MLP
        baseline_targets, _ = resolve_targets(self.model(("fc1", "fc2")), "baseline")
        self.assertEqual(baseline_targets, list(BASELINE_TARGETS))  # the default path is untouched

    @staticmethod
    def kimi_model(vision_blocks=3, layers=3, experts=4):
        import torch
        nn = torch.nn
        with torch.device("meta"):
            model = nn.Module()
            model.vision_tower = nn.Module()
            model.vision_tower.encoder = nn.Module()
            model.vision_tower.encoder.blocks = nn.ModuleList()
            for _ in range(vision_blocks):
                block = nn.Module()
                block.wqkv, block.wo = nn.Linear(8, 24), nn.Linear(8, 8)
                block.mlp = nn.Module()
                block.mlp.fc0, block.mlp.fc1 = nn.Linear(8, 16), nn.Linear(16, 8)
                model.vision_tower.encoder.blocks.append(block)
            model.language_model = nn.Module()
            model.language_model.model = nn.Module()
            model.language_model.model.layers = nn.ModuleList()
            def mlp():
                module = nn.Module()
                for name in ("gate_proj", "up_proj", "down_proj"):
                    setattr(module, name, nn.Linear(8, 8))
                return module
            for index in range(layers):
                layer = nn.Module()
                layer.self_attn = nn.Module()
                for name in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                    setattr(layer.self_attn, name, nn.Linear(8, 8))
                if index == 0:
                    layer.mlp = mlp()  # the first layer is dense
                else:
                    layer.mlp = nn.Module()
                    layer.mlp.experts = nn.ModuleList(mlp() for _ in range(experts))
                    layer.mlp.shared_experts = mlp()
                model.language_model.model.layers.append(layer)
        return model

    def test_kimi_adapts_vision_attention_and_shared_mlps_but_not_routed_experts(self):
        model = self.kimi_model()
        targets, audit = resolve_targets(model, "kimi", 4)
        self.assertFalse(any(".experts." in n for n in targets))
        self.assertEqual(audit["baseline_modules"], [])
        # 3 blocks x 4 vision + 3 layers x 4 attention + 3 dense + 2 layers x 3 shared.
        self.assertEqual(len(targets), 12 + 12 + 3 + 6)
        baseline_targets, _ = resolve_targets(model, "baseline", 4)  # suffixes would hit experts
        self.assertEqual(baseline_targets, list(BASELINE_TARGETS))

    def test_kimi_fails_closed_on_a_missing_projection(self):
        model = self.kimi_model()
        del model.language_model.model.layers[1].self_attn.kv_b_proj
        with self.assertRaisesRegex(ValueError, "layers.1 lacks"):
            resolve_targets(model, "kimi")
        with self.assertRaisesRegex(ValueError, "No Kimi-VL"):
            resolve_targets(self.model(), "kimi")

    def test_vision_mlp_fails_without_a_vision_mlp(self):
        model = self.model(())
        with self.assertRaisesRegex(ValueError, "No vision MLP"):
            resolve_targets(model, "vision-mlp")

    def test_wrong_architecture_fails_closed(self):
        import torch
        model = self.model()
        model.visual.blocks[0].attn.proj = torch.nn.Conv3d(3, 3, 1)
        with self.assertRaisesRegex(ValueError, "not Linear"):
            resolve_targets(model, "vision-attention")
        del model.visual.blocks[-1]
        with self.assertRaisesRegex(ValueError, "64 visual"):
            resolve_targets(model, "vision-attention")

    @unittest.skipUnless(importlib.util.find_spec("peft"), "PEFT optional")
    def test_peft_actual_target_set_and_parameter_count(self):
        from peft import LoraConfig, get_peft_model
        model = self.model()
        targets, audit = resolve_targets(model, "vision-attention")
        adapted = get_peft_model(model, LoraConfig(r=16, target_modules=targets))
        actual = [n.removeprefix("base_model.model.") for n, m in adapted.named_modules() if hasattr(m, "lora_A")]
        self.assertEqual(set(actual), set(targets))
        self.assertEqual(sum(p.numel() for p in adapted.parameters() if p.requires_grad), audit["expected_adapter_parameters"])

    @unittest.skipUnless(importlib.util.find_spec("peft"), "PEFT optional")
    def test_gradient_audit_accepts_live_B_and_rejects_zero_missing_nonfinite(self):
        import torch
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        model = get_peft_model(model, LoraConfig(r=2, target_modules=["0"]))
        audit = AttentionGradientAudit(model, ["0"])
        self.assertFalse(audit.report()["passed"])
        model(torch.ones(2, 4)).sum().backward()
        self.assertTrue(audit.report()["passed"])
        audit.close()
        audit = AttentionGradientAudit(model, ["0"])
        (model(torch.ones(2, 4)).sum() * 0).backward()
        self.assertFalse(audit.report()["passed"])
        model(torch.ones(2, 4)).sum().mul(float("nan")).backward()
        self.assertFalse(audit.report()["passed"])
        audit.close()
