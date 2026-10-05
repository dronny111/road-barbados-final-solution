# Qwen3-VL-4B with vision-MLP LoRA, fold 0, Kaggle (two T4s)

The Kaggle build of `../runs_qwen3vl4b_vmlp_colab/`: the same recipe, hypothesis and gates
(alone < 0.11409; vote with 7B fp16 + NF4 < 0.10991). It differs only in placement:
`GPU_INDEX = "0,1"` shards the model across Kaggle's two T4s. Training the vision layers
made the 2B run out of memory on one T4 without checkpointing, and this model is twice as
large. The 2B vision-MLP run took the old 2B from 0.17458 to 0.11440, so the first 4B run
(0.16747, vision frozen) is expected to improve sharply. About 5 h. Account: `<your-kaggle-user>`.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_qwen3vl4b_vmlp_kaggle/config.json --output baselines/vlm/runs_qwen3vl4b_vmlp_kaggle/kaggle_qwen3vl4b_vmlp_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_qwen3vl4b_vmlp_kaggle --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels output <your-kaggle-user>/road-qwen3vl4b-vmlp-fold0 -p experiments/runs/qwen3vl4b_vmlp_small \
  --file-pattern '(validation_predictions|validation_reference|test_predictions|run_config|submission|commands|lora_audit)\.(csv|json|jsonl)$'
```
