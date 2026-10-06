PYTHON ?= python3
PYTHONPATH := src

.PHONY: test preflight folds score submission-check verify-submission notebook clean-images final-ensemble figures demo

# KMP_DUPLICATE_LIB_OK avoids "OMP Error #15" when two OpenMP runtimes are loaded (macOS).
test:
	KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

# Needs Train.csv, Test.csv, SampleSubmission.csv and images/ in the repository root (not shipped).
preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/preflight.py --full-images

# Regenerates data/splits/folds.csv; the pinned FOLD_SHA256 in src/road_ocr/trocr_support.py must match.
folds:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_folds.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from road_ocr.trocr_support import verify_folds; verify_folds('data/splits/folds.csv'); print('fold manifest matches FOLD_SHA256')"

score:
	@test -n "$(REF)" || (echo "REF is required" && exit 2)
	@test -n "$(PRED)" || (echo "PRED is required" && exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/score.py --reference "$(REF)" --predictions "$(PRED)"

submission-check:
	@test -n "$(FILE)" || (echo "FILE is required" && exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_submission.py "$(FILE)"

# Needs Test.csv and the rebuilt submission/20261004_allvote10_hill.csv.
verify-submission:
	cd submission && shasum -a 256 -c SHA256SUMS
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_submission.py submission/20261004_allvote10_hill.csv

# Build a Kaggle notebook for one member:
#   make notebook CONFIG=baselines/vlm/runs_2b_vmlp_softkd_fold0_kaggle/config.json OUT=softkd_fold0.ipynb
notebook:
	@test -n "$(CONFIG)" || (echo "CONFIG is required" && exit 2)
	@test -n "$(OUT)" || (echo "OUT is required" && exit 2)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_kaggle_notebook.py --config "$(CONFIG)" --output "$(OUT)"

# Cleaned images (members 6, 8, 9): PNGs in images_clean/, Kaggle-ready JPEGs in images_clean_jpg/.
clean-images:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/clean_images.py --out images_clean --reencode-jpeg images_clean_jpg

# Rebuild the submitted file from the ten members' prediction files (docs/REPRODUCE.md). DRY=1 prints the commands.
final-ensemble:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_final_ensemble.py $(if $(DRY),--dry-run)

figures:
	$(PYTHON) scripts/make_figures.py

demo:
	$(PYTHON) scripts/make_demo.py
