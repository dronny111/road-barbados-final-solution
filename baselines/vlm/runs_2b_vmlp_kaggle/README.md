# Qwen2-VL-2B with vision-MLP LoRA, fold 0, Kaggle T4

The Kaggle build of `../runs_2b_vmlp_colab/`. It uses the same config file
(`../runs_2b_vmlp_colab/config.json`), so the recipe, hypothesis and gates are identical: see
that README. Only the runtime differs.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_2b_vmlp_colab/config.json --output baselines/vlm/runs_2b_vmlp_kaggle/kaggle_2b_vmlp_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_2b_vmlp_kaggle --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels output <your-kaggle-user-2>/road-2b-vmlp-fold0 -p experiments/runs/2b_vmlp_small \
  --file-pattern '(validation_predictions|validation_reference|test_predictions|run_config|submission|commands)\.(csv|json|jsonl)$'
```
