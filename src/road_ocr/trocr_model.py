"""Load a TrOCR (VisionEncoderDecoder) checkpoint that was trained at a non-square input size.

`Kansallisarkisto/multicentury-htr-model` keeps the 384x384 ViT config of its
`microsoft/trocr-large-handwritten` base, but its processor resizes lines to 192x1024. ViT
then has to interpolate its position embeddings, and its patch embedding must skip the fixed
size check. The model card monkeypatches both; ``interpolate_pos_encoding=True`` does the
same through the public argument, so every encoder call (training, generate) passes it.
"""
from __future__ import annotations

from pathlib import Path


def allow_any_input_size() -> None:
    """Make every ViT embedding call interpolate position encodings. Idempotent."""
    from transformers.models.vit import modeling_vit

    embeddings = modeling_vit.ViTEmbeddings
    if getattr(embeddings.forward, "_road_any_size", False):
        return
    original = embeddings.forward

    def forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None, **kwargs):
        return original(self, pixel_values, bool_masked_pos=bool_masked_pos,
                        interpolate_pos_encoding=True, **kwargs)

    forward._road_any_size = True
    embeddings.forward = forward


def roberta_tokenizer_from_files(model_dir):
    """Build TrOCR's RoBERTa byte-level BPE tokenizer from vocab.json + merges.txt.

    Older checkpoints (e.g. microsoft/trocr-base-handwritten) ship no tokenizer.json, and
    transformers 5 cannot convert their slow tokenizer without extra packages. This is the
    standard RoBERTa construction; it produced identical token ids to the multicentury
    model's tokenizer.json on real lines (their vocab files are byte-identical).
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import RobertaTokenizerFast

    model_dir = Path(model_dir)
    backend = Tokenizer(models.BPE.from_file(str(model_dir / "vocab.json"), str(model_dir / "merges.txt")))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.RobertaProcessing(
        ("</s>", backend.token_to_id("</s>")), ("<s>", backend.token_to_id("<s>")),
        add_prefix_space=False, trim_offsets=True)
    return RobertaTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                sep_token="</s>", cls_token="<s>", unk_token="<unk>",
                                pad_token="<pad>", mask_token="<mask>")


def load_trocr(model_dir, device, dtype=None):
    """Return (processor, model) for a local TrOCR checkpoint directory."""
    from transformers import AutoImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel

    allow_any_input_size()
    if (Path(model_dir) / "tokenizer.json").is_file():
        processor = TrOCRProcessor.from_pretrained(model_dir)
    else:
        processor = TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(model_dir),
                                   tokenizer=roberta_tokenizer_from_files(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    # The decoder shifts labels right with these, and generate() reads them from the
    # generation config; a checkpoint missing them trains or decodes garbage. Older
    # configs (trocr-base-handwritten) omit the attribute entirely, hence getattr.
    start = getattr(model.config, "decoder_start_token_id", None)
    start = tokenizer.cls_token_id if start is None else start
    for config in (model.config, model.generation_config):
        config.decoder_start_token_id = start
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.sep_token_id
    if dtype is not None:
        model = model.to(dtype)
    return processor, model.to(device)
