# LLM Fine-Tuning Lab

Domain-adapting an open LLM into a bilingual (English / Urdu) customer-support
assistant with LoRA. Built like an enterprise procedure, not a notebook: every
stage is documented, every number comes from a real run.

## The problem

Pakistani support desks serve customers in English, Roman Urdu, and Urdu
script. Off-the-shelf chat models answer well in English but poorly in the
other two. This project teaches a small open model the domain: support
answers with the right structure, in the customer's own language.

## How it was done (the procedure)

| Stage | Doc | What happened |
|---|---|---|
| 1. Problem brief | [docs/01-problem-brief.md](docs/01-problem-brief.md) | Business problem, success criteria, non-goals |
| 2. Data pipeline | [docs/02-data-pipeline.md](docs/02-data-pipeline.md) | 45 hand-written scenarios x 3 languages, deterministic expansion, 480 train / 60 held-out |
| 3. Experiment plan | [docs/03-experiment-plan.md](docs/03-experiment-plan.md) | Hypotheses, LoRA config, eval gates (internal RFC style) |
| 4. Training | [docs/04-training-log.md](docs/04-training-log.md) | Real run: config, hardware, loss curve |
| 5. Evaluation | [docs/05-eval-report.md](docs/05-eval-report.md) | Before/after on 60 unseen examples |
| 6. Deployment | [docs/06-deployment-readiness.md](docs/06-deployment-readiness.md) | How this ships: serving, monitoring, rollback |

## The run (run-01)

- **Base model:** Qwen2.5-0.5B-Instruct
- **Method:** LoRA (r=16, alpha=32) via PEFT — 8,798,208 trainable params (1.75% of 502.8M)
- **Data:** 480 bilingual support examples, 2 epochs, CPU-only box
- **Training time:** 58.4 min (final invocation; see docs/04 for the full attempt log) | **Final train loss:** 0.4778

## Results (held-out set, 60 examples the model never saw)

| Metric | Base model | Fine-tuned | Change |
|---|---|---|---|
| Answers in customer's language | 76.7% (46/60) | 100% (60/60) | +23.3 pts |
| Domain keyword coverage | 0.059 | 0.067 | +0.008 |

Full before/after generations: [docs/05-eval-report.md](docs/05-eval-report.md).

## Evidence

![dataset preview](docs/screenshots/01-dataset-preview.png)
![dataset stats](docs/screenshots/02-dataset-stats.png)
![training terminal](docs/screenshots/03-training-terminal.png)
![loss curve](docs/screenshots/04-loss-curve.png)
![eval results](docs/screenshots/05-eval-comparison.png)
![inference demo](docs/screenshots/06-inference-demo.png)

## Run it yourself

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. build the dataset (deterministic, seed 42)
python3 data/generate_dataset.py

# 2. train (CPU-friendly; see configs/train.yaml)
.venv/bin/python training/train.py

# 3. evaluate base vs tuned on the held-out set
.venv/bin/python eval/evaluate.py

# 4. ask the tuned model a question
.venv/bin/python inference/generate.py --question "Mera bill kab due hai?"

# 5. serve it
uvicorn inference.app:app --host 0.0.0.0 --port 8000
```

## Honest notes

- The dataset is constructed (hand-written scenarios), not scraped. No real
  customer data exists in this project.
- run-01 trained on a CPU-only machine (2 CPUs, 8 GB RAM, no GPU). The config
  was sized for that hardware. A GPU run with QLoRA is listed as a follow-up
  in docs/06.
- Eval metrics are simple directional heuristics on 60 examples. The full
  generations are published so anyone can judge for themselves.
