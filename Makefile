# EntitlementShield — developer tasks.
# Phase 1 build plan: docs/phase-1-plan.md. Only `test` must be green in Step 0;
# seed/detector/eval are placeholders until the steps that implement them.

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Targets:"
	@echo "  install    create .venv and install the project with dev extras"
	@echo "  test       run the pytest suite"
	@echo "  lint       ruff check"
	@echo "  typecheck  mypy"
	@echo "  seed       (Step 2) generate the dev-tier world"
	@echo "  detector   (Step 6) run the batch detector over the dev world"
	@echo "  release-escalations (P3.6) release expired L-5 escalations"
	@echo "  eval       (Step 7) run the evaluation harness + ablation"
	@echo "  arm3       (P2.3) run the logistic re-ranker (attribution control, dev tier)"
	@echo "  serve      (P2.4) run the API + analyst console (uvicorn, dev tier)"
	@echo "  eval-json  (P2.4) write the dev eval report the console's evaluation page reads"
	@echo "  load-tier2 (P2.1) build the Tier 2 inference + GT stores from the parquets"
	@echo "  republish-tier3 (Tier 3) rebuild+atomically republish the Tier 3 pair (correct tier3 provenance)"
	@echo "  classify-families (P2.2) write docs/family-mapping.md from the Tier 2 world"
	@echo "  manifest       (Step 8) write config/feature-manifest.json from live config"
	@echo "  manifest-check (Step 8) verify the committed manifest matches regeneration"
	@echo "  tune           (Step 8) run the deterministic weight-tuning driver + ledger"
	@echo "  tune-check     (Step 8) verify the committed tuning ledger reproduces"
	@echo "  reviewer-manifest       (P3.9) write config/reviewer-manifest.json from live config"
	@echo "  reviewer-manifest-check (P3.9) verify the committed reviewer manifest matches"
	@echo "  arm4           (P3.9) score the LLM reviewer arms 4/4b/2 on fresh dev worlds"
	@echo "  demo           (Phase 5 Track 2) reseed dev_arm4, eval-json, pre-run reviewer, print case URLs, serve"

$(VENV):
	python3 -m venv $(VENV)

.PHONY: install
install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: test
test:
	$(VENV)/bin/pytest

.PHONY: lint
lint:
	$(VENV)/bin/ruff check backend scripts

.PHONY: typecheck
typecheck:
	$(VENV)/bin/mypy
	$(VENV)/bin/mypy scripts

.PHONY: seed
seed:
	$(PY) scripts/seed.py

.PHONY: load-tier2
load-tier2:
	$(PY) scripts/load_tier2.py

.PHONY: republish-tier3
republish-tier3:  ## (Tier 3) explicit opt-in rebuild + atomic republish of the Tier 3 pair with tier3 provenance
	$(PY) scripts/republish_tier3.py

.PHONY: classify-families
classify-families:  ## (P2.2) classify Tier 2 families → docs/family-mapping.md (needs `make load-tier2`)
	$(PY) scripts/classify_families.py --from-stores

.PHONY: detector
detector:
	$(PY) scripts/detector_run.py

.PHONY: release-escalations
release-escalations:  ## (P3.6) release expired L-5 escalations (also run nightly by `detector`)
	$(PY) scripts/release_escalations.py

.PHONY: eval
eval:
	$(PY) scripts/eval_run.py

.PHONY: ablation
ablation:
	$(PY) scripts/ablation_run.py

.PHONY: eval-json
eval-json:  ## (P2.4) write the dev-tier eval report for the console (needs `make seed`)
	$(PY) scripts/eval_run.py --json data/synthetic/eval-report.json

.PHONY: serve
serve:  ## (P2.4) run the API + analyst console on :8000 (needs `make seed`)
	$(VENV)/bin/uvicorn app.api.app:create_app --factory --app-dir backend --reload

.PHONY: arm3
arm3:  ## (P2.3) run the arm-3 logistic re-ranker on the seeded dev world (needs `make seed`)
	$(PY) scripts/arm3_run.py

.PHONY: arm4
arm4:  ## (P3.9) score reviewer arms 4/4b/2 on freshly seeded dev worlds (dev tier, W-1)
	$(PY) scripts/arm4_run.py --arm both

DEMO_PROVIDER_CONFIG := $(CURDIR)/config/reviewer-provider.demo.yaml
DEMO_DIR := $(CURDIR)/data/synthetic/demo
DEMO_DATABASE_URL := sqlite:///$(DEMO_DIR)/inference.db
DEMO_GROUND_TRUTH_DATABASE_URL := sqlite:///$(DEMO_DIR)/ground_truth.db
DEMO_EVAL_REPORT := $(DEMO_DIR)/eval-report.json
DEMO_PICKS := $(DEMO_DIR)/picks.json
DEMO_ARTIFACT_DIR := $(DEMO_DIR)/artifacts
DEMO_PORT ?= 8001
DEMO_BASE_URL := http://localhost:$(DEMO_PORT)
DEMO_ENV = DATABASE_URL=$(DEMO_DATABASE_URL) GROUND_TRUTH_DATABASE_URL=$(DEMO_GROUND_TRUTH_DATABASE_URL) EVAL_REPORT_PATH=$(DEMO_EVAL_REPORT) WORLD_SEED=1

.PHONY: demo
demo:  ## (Phase 5 Track 2) build an isolated dev_arm4 demo world, pre-run advice, serve
	@echo "[demo] provider: forcing config/reviewer-provider.demo.yaml via"
	@echo "[demo]   REVIEWER_PROVIDER_CONFIG_PATH (the committed config/reviewer-provider.yaml"
	@echo "[demo]   'active: openai_compatible' targets the local LLM on :8000; the"
	@echo "[demo]   console serves independently on :$(DEMO_PORT)."
	@echo "[demo]   Demo config is 'active: typesafe' (Jev, P5.4) for real typed answers"
	@echo "[demo]   -- set 'active: fake' in that file instead for a network-free run."
	@if grep -qE '^active:[[:space:]]*typesafe' $(DEMO_PROVIDER_CONFIG) \
		&& [ -z "$$TYPESAFE_API_KEY" ] \
		&& ! grep -qE '^TYPESAFE_API_KEY=.+' .env 2>/dev/null; then \
		echo "[demo] ERROR: config/reviewer-provider.demo.yaml is 'active: typesafe' but" >&2; \
		echo "[demo]   TYPESAFE_API_KEY is not set (env or .env). Export it, or switch that" >&2; \
		echo "[demo]   file to 'active: fake' for a network-free demo, then re-run 'make demo'." >&2; \
		exit 1; \
		fi
	@if grep -qE '^active:[[:space:]]*anthropic' $(DEMO_PROVIDER_CONFIG) \
		&& [ -z "$$ANTHROPIC_API_KEY" ] \
		&& ! grep -qE '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null; then \
		echo "[demo] ERROR: config/reviewer-provider.demo.yaml is 'active: anthropic' but" >&2; \
		echo "[demo]   ANTHROPIC_API_KEY is not set (env or .env). Set it, or switch that" >&2; \
		echo "[demo]   file to 'active: fake' for a network-free demo, then re-run 'make demo'." >&2; \
		exit 1; \
		fi
	@REVIEWER_PROVIDER_CONFIG_PATH=$(DEMO_PROVIDER_CONFIG) $(PY) -c \
		'from app.providers.factory import load_provider_config; load_provider_config()'
	@echo "[demo] rebuilding isolated stores under $(DEMO_DIR) (ordinary dev stores untouched)"
	mkdir -p $(DEMO_DIR)
	rm -f $(DEMO_DIR)/inference.db $(DEMO_DIR)/ground_truth.db $(DEMO_EVAL_REPORT) $(DEMO_PICKS)
	$(DEMO_ENV) $(PY) scripts/seed.py --preset dev_arm4 --artifact-dir $(DEMO_ARTIFACT_DIR)
	$(DEMO_ENV) $(PY) scripts/eval_run.py --json $(DEMO_EVAL_REPORT)
	@echo "[demo] running the detector so the console's case queue is populated..."
	$(DEMO_ENV) $(PY) scripts/detector_run.py
	$(DEMO_ENV) REVIEWER_PROVIDER_CONFIG_PATH=$(DEMO_PROVIDER_CONFIG) $(PY) scripts/demo_pick.py --base-url $(DEMO_BASE_URL) --out $(DEMO_PICKS)
	@echo "[demo] pre-running advice-only reviewer samples so the narrative panel is"
	@echo "[demo]   already populated on first page load..."
	$(DEMO_ENV) REVIEWER_PROVIDER_CONFIG_PATH=$(DEMO_PROVIDER_CONFIG) $(PY) scripts/demo_review.py --picks $(DEMO_PICKS)
	@echo "[demo] starting the console on $(DEMO_BASE_URL) (ctrl-c to stop)..."
	$(DEMO_ENV) REVIEWER_PROVIDER_CONFIG_PATH=$(DEMO_PROVIDER_CONFIG) $(VENV)/bin/uvicorn app.api.app:create_app --factory --app-dir backend --reload --port $(DEMO_PORT)

.PHONY: manifest
manifest:
	$(PY) scripts/write_manifest.py

.PHONY: manifest-check
manifest-check:
	$(PY) scripts/write_manifest.py --check

.PHONY: reviewer-manifest
reviewer-manifest:
	$(PY) scripts/write_reviewer_manifest.py

.PHONY: reviewer-manifest-check
reviewer-manifest-check:
	$(PY) scripts/write_reviewer_manifest.py --check

.PHONY: tune
tune:
	$(PY) scripts/tune_weights.py

.PHONY: tune-check
tune-check:
	$(PY) scripts/tune_weights.py --check
