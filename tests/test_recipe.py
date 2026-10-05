import importlib.util
import random
import sys
import unittest

sys.path.insert(0, "src")
from road_ocr.recipe import curriculum_order, fork_adapter, stratum, use_alt_view  # noqa: E402

META = ([dict(pseudo=False, height=h, length=n, support=1.0) for h, n in
         [(60, 30), (60, 90), (400, 50), (70, 20), (500, 70)]]
        + [dict(pseudo=True, height=h, length=40, support=s) for h, s in [(60, 0.7), (60, 0.9), (400, 0.8)]])


class RecipeTest(unittest.TestCase):
    def test_curriculum_is_a_deterministic_permutation_easy_first_pseudo_last(self):
        order = curriculum_order(META, seed=1)
        self.assertEqual(sorted(order), list(range(len(META))))
        self.assertEqual(order, curriculum_order(META, seed=1))
        kinds = [(META[i]["pseudo"], stratum(META[i]["height"])) for i in order]
        self.assertEqual(kinds[:3], [(False, "tight")] * 3)         # real tight first
        self.assertEqual({k[0] for k in kinds[-3:]}, {True})        # pseudo last
        self.assertEqual([META[i]["support"] for i in order[-3:]], [0.9, 0.8, 0.7])  # by support

    def test_alt_view_is_seeded_and_off_without_rng(self):
        self.assertFalse(use_alt_view(None, 1.0))
        a = [use_alt_view(random.Random(5), 0.5) for _ in range(3)]
        self.assertEqual(a, [use_alt_view(random.Random(5), 0.5) for _ in range(3)])

    @unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("torch"), "peft optional")
    def test_fork_copies_the_shared_adapter_into_the_expert(self):
        import torch
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=2, target_modules=["lin"])
        base = torch.nn.Module()
        base.lin = torch.nn.Linear(4, 4)
        model = get_peft_model(base, cfg)
        model.add_adapter("loose", cfg)
        for p in model.parameters():
            if p.requires_grad:
                p.data.normal_()
        before = fork_adapter(model)
        params = dict(model.named_parameters())
        self.assertGreater(before, 0)
        for n, p in params.items():
            if "lora_A.default" in n:
                self.assertTrue(torch.equal(p, params[n.replace(".default.", ".loose.")]))


if __name__ == "__main__":
    unittest.main()
