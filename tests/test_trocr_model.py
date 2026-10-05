import unittest


class AnyInputSizeTest(unittest.TestCase):
    def test_square_vit_config_trains_and_generates_on_a_wide_input(self):
        try:
            import torch
            from transformers import (TrOCRConfig, TrOCRForCausalLM, ViTConfig, ViTModel,
                                      VisionEncoderDecoderModel)
        except ImportError:
            self.skipTest("optional VLM environment test")
        from road_ocr.trocr_model import allow_any_input_size

        allow_any_input_size()
        allow_any_input_size()  # idempotent: patching twice must not stack wrappers
        encoder = ViTModel(ViTConfig(image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32))
        decoder = TrOCRForCausalLM(TrOCRConfig(vocab_size=20, d_model=16, decoder_layers=1,
                                               decoder_attention_heads=2, decoder_ffn_dim=32,
                                               cross_attention_hidden_size=16))
        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.decoder_start_token_id, model.config.pad_token_id, model.config.eos_token_id = 0, 1, 2
        wide = torch.randn(2, 3, 16, 64)  # 2x8 patches against a 4x4 position grid
        loss = model(pixel_values=wide, labels=torch.tensor([[3, 4, 2], [5, 2, 1]])).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        out = model.generate(wide, max_new_tokens=3)
        self.assertEqual(out.shape[0], 2)


class TokenizerFallbackTest(unittest.TestCase):
    def test_builds_a_roberta_tokenizer_when_tokenizer_json_is_missing(self):
        try:
            import tokenizers  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("optional VLM environment test")
        import json
        import tempfile
        from pathlib import Path
        from road_ocr.trocr_model import roberta_tokenizer_from_files

        # Byte-level BPE: 'Ġ' marks a leading space. A tiny vocab spelling "the said".
        vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "t": 4, "h": 5, "e": 6, "Ġ": 7,
                 "s": 8, "a": 9, "i": 10, "d": 11, "th": 12, "the": 13, "Ġs": 14, "<mask>": 15}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "vocab.json").write_text(json.dumps(vocab))
            Path(tmp, "merges.txt").write_text("#version: 0.2\nt h\nth e\n\u0120 s\n")
            tok = roberta_tokenizer_from_files(tmp)
            ids = tok("the said").input_ids
            self.assertEqual((ids[0], ids[-1]), (0, 2))  # wrapped in <s> ... </s>
            self.assertEqual(tok.decode(ids, skip_special_tokens=True), "the said")
            self.assertEqual(tok.pad_token_id, 1)


if __name__ == "__main__":
    unittest.main()
