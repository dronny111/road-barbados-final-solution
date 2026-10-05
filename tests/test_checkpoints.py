import json
import tempfile
import unittest
from pathlib import Path

from road_ocr.checkpoints import validate_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_weights_only_can_be_evaluated_but_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
            (path / "trainer_state.json").write_text(json.dumps({"global_step": 500}))
            (path / "adapter_model.safetensors").write_bytes(b"synthetic test fixture")
            self.assertEqual(validate_checkpoint(path)["global_step"], 500)
            with self.assertRaisesRegex(ValueError, "optimizer.pt"):
                validate_checkpoint(path, resume=True)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).write_bytes(b"synthetic test fixture")
            self.assertEqual(validate_checkpoint(path, resume=True)["global_step"], 500)
            (path / "adapter_model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "adapter_model"):
                validate_checkpoint(path)

    def test_incomplete_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
            (path / "trainer_state.json").write_text("{}")
            (path / "adapter_model.safetensors").write_bytes(b"synthetic test fixture")
            with self.assertRaisesRegex(ValueError, "global step"):
                validate_checkpoint(path)
