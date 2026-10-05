"""Optional, tiny CPU integration test; no downloaded model or competition data.

ROAD_TEST_VLM_RESUME=1 PYTHONPATH=src .venv-vlm/bin/python -m unittest discover \
    -s tests -p test_vlm_resume.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from road_ocr.checkpoints import validate_checkpoint


@unittest.skipUnless(os.environ.get("ROAD_TEST_VLM_RESUME") == "1", "optional VLM environment test")
class ResumeIntegrationTest(unittest.TestCase):
    def test_resumed_lora_matches_uninterrupted(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, set_seed

        torch.set_num_threads(1)
        data = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]] * 8,
            "labels": [[1, 2, 3, 4, 5, 6, 7, 8]] * 8,
        })

        def model():
            set_seed(123)
            base = GPT2LMHeadModel(GPT2Config(
                vocab_size=16, n_positions=16, n_embd=16, n_layer=1,
                n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            ))
            return get_peft_model(base, LoraConfig(
                task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["c_attn"],
                lora_dropout=0.05, fan_in_fan_out=True,
            ))

        def trainer(member, output):
            return Trainer(model=member, args=TrainingArguments(
                output_dir=str(output), use_cpu=True, max_steps=4,
                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                learning_rate=1e-3, save_strategy="steps", save_steps=2,
                report_to=[], seed=123, disable_tqdm=True,
            ), train_dataset=data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = model()
            trainer(full, root / "original").train()
            checkpoint = root / "original" / "checkpoint-2"
            self.assertEqual(validate_checkpoint(checkpoint, resume=True)["global_step"], 2)
            resumed = model()
            resumed_trainer = trainer(resumed, root / "resumed")
            resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed_trainer.state.global_step, 4)
            for key, tensor in get_peft_model_state_dict(full).items():
                torch.testing.assert_close(
                    tensor, get_peft_model_state_dict(resumed)[key], rtol=0, atol=0,
                )
            self.assertTrue(checkpoint.is_dir())
