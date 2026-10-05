import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
import math  # noqa: E402
from road_ocr.ctc_aux import (CtcHead, ReconHead, alignment_entropy, ctc_loss,  # noqa: E402
                              infeasible_fraction, pool_frames, recon_loss)


class CtcAuxTest(unittest.TestCase):
    def test_shape_and_finite_loss(self):
        head = CtcHead(8, 20)
        hidden = torch.randn(2, 1 + 3 * 6, 8)  # CLS + 3x6 grid
        self.assertEqual(head(hidden, 3, 6).shape, (2, 6, 21))
        labels = torch.tensor([[0, 5, 6, 6, 2], [0, 7, 2, -100, -100]])
        loss = ctc_loss(head, hidden, labels, 3, 6, [0, 2])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_infeasible_row_is_zeroed_not_nan(self):
        head = CtcHead(8, 20)
        labels = torch.tensor([[0] + [5] * 10 + [2]])  # 10 targets > 6 frames
        loss = ctc_loss(head, torch.randn(1, 18, 8), labels, 3, 6, [0, 2])
        self.assertEqual(float(loss), 0.0)

    def test_infeasible_fraction(self):
        self.assertEqual(infeasible_fraction([3, 70, 64], 64), 1 / 3)


    def test_pool_frames_drops_cls_and_averages_height(self):
        hidden = torch.cat([torch.full((1, 1, 2), 99.0), torch.arange(12.0).reshape(1, 6, 2)], dim=1)
        frames = pool_frames(hidden, 2, 3)  # rows [0..2] and [3..5] of the 2x3 grid
        self.assertTrue(torch.equal(frames, (hidden[:, 1:4] + hidden[:, 4:7]) / 2))


class ReconAlignTest(unittest.TestCase):
    def test_recon_shape_finite_and_reaches_encoder(self):
        torch.manual_seed(0)
        head, embed = ReconHead(8, frames=6, heads=2), torch.nn.Embedding(20, 8)
        hidden = torch.randn(2, 1 + 3 * 6, 8, requires_grad=True)
        labels = torch.tensor([[0, 5, 6, 2], [0, 7, 2, -100]])
        self.assertEqual(head(embed(labels.clamp_min(0)), labels == -100).shape, (2, 6, 8))
        loss = recon_loss(head, embed, hidden, labels, 3, 6, pad_id=1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(hidden.grad.abs().sum()), 0)  # target is not detached

    def test_entropy_uniform_onehot_and_mask(self):
        s = 10
        uniform = torch.full((1, 2, 3, s), 1 / s)
        labels = torch.tensor([[5, 6, -100]])
        self.assertAlmostEqual(float(alignment_entropy(uniform, labels)), math.log(s), places=5)
        onehot = torch.zeros(1, 2, 3, s)
        onehot[..., 0] = 1
        self.assertAlmostEqual(float(alignment_entropy(onehot, labels)), 0.0, places=6)
        mixed = onehot.clone()
        mixed[:, :, 2] = 1 / s  # the masked token is uniform; it must not count
        self.assertAlmostEqual(float(alignment_entropy(mixed, labels)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
