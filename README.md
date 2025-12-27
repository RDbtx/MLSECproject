# MLSECproject
Repo for Machine Learning Security exam project

## Project 1 — To‑Do Checklist (RobustBench + AutoAttack)

Goal: **Re-evaluate 5 RobustBench CIFAR-10 (L∞) models with AutoAttack** on a **100–200 image subset**, sweeping **ε**, and study how the **ranking** changes with ε.

---

## 0) Decide the experiment spec (write this down first)
- [ ] **Pick 5 models** from RobustBench: dataset=`cifar10`, threat_model=`Linf`
- [ ] Prefer a mix of strong + medium models (so ranking can change)
- [ ] Choose **subset size**: `n_examples ∈ {100, 200}`
- [ ] Choose **ε sweep** (example): `{1, 2, 4, 8, 12, 16} / 255`
- [ ] Choose batch size `bs` (example): 50 (adjust to GPU memory)
- [ ] Fix random seed(s) and record hardware + software versions

Deliverables you should plan to produce:
- [ ] Table: robust accuracy for each model at each ε
- [ ] Plot(s): robust accuracy vs ε (lines per model)
- [ ] Ranking analysis: ranks per ε + rank correlation / rank flips
- [ ] Short write-up: methods, results, discussion, limitations

---

## 1) Environment setup
- [ ] Create a clean environment (venv/conda)
- [ ] Install dependencies
  - [ ] `torch` (CUDA build if available)
  - [ ] `robustbench`
  - [ ] `autoattack`
  - [ ] `numpy`, `pandas`, `matplotlib`
  - [ ] (optional) `scipy` for Spearman correlation

Example (pip):
- [ ] `pip install torch torchvision torchaudio` (choose the right CUDA/CPU wheel)
- [ ] `pip install robustbench autoattack numpy pandas matplotlib scipy`

Sanity checks:
- [ ] `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `python -c "import robustbench, autoattack; print('ok')"`

---

## 2) Repository / folder structure
- [ ] Create a small repo/folder, e.g.
  - [ ] `src/`
  - [ ] `results/`
  - [ ] `figures/`
  - [ ] `README.md`
- [ ] Add a `config.py` or `config.yaml` (models, eps_list, subset size, bs, seed)

---

## 3) Implement the evaluation script
Create a script (e.g., `src/run_autoattack_sweep.py`) that:

### 3.1 Load data (RobustBench helper)
- [ ] Load CIFAR-10 test set subset using RobustBench utilities
- [ ] Move tensors to `device` (`cuda` if available)
- [ ] Store the chosen indices (for reproducibility)
  - [ ] Save indices to `results/subset_indices.npy`

### 3.2 Load 5 models (RobustBench)
- [ ] For each model name:
  - [ ] `load_model(model_name=..., dataset="cifar10", threat_model="Linf")`
  - [ ] `model.eval()` and move to device
  - [ ] (optional) run a clean accuracy check on the subset

### 3.3 Run AutoAttack for each ε
- [ ] For each model and each ε:
  - [ ] Instantiate `AutoAttack(model, norm="Linf", eps=eps, version="standard")`
  - [ ] Run: `run_standard_evaluation(x, y, bs=bs)`
  - [ ] Compute robust accuracy on the returned adversarial examples
  - [ ] Save results in a structured format (dict → CSV/JSON)

**Tip:** Save intermediate results after each ε so you don’t lose work if something crashes.

---

## 4) Save results cleanly
- [ ] Save a **CSV**: rows = models, columns = eps, values = robust accuracy
  - [ ] `results/robust_accuracy.csv`
- [ ] Save a **JSON** with metadata:
  - [ ] model names, eps list, subset size, batch size, seed
  - [ ] torch/robustbench/autoattack versions
  - [ ] device info (GPU model if any)
  - [ ] `results/metadata.json`

---

## 5) Analysis: ranking vs ε
Create an analysis notebook or script (e.g., `src/analyze_rankings.py`) that:

### 5.1 Rankings per ε
- [ ] For each ε:
  - [ ] Rank models by robust accuracy (descending)
  - [ ] Save ranking table: `results/rankings.csv`

### 5.2 Quantify rank stability
Pick at least one:
- [ ] **Spearman rank correlation** between rankings at εᵢ and εⱼ
- [ ] **Kendall τ** correlation (optional alternative)
- [ ] Count **rank flips** between adjacent ε values
- [ ] Compute “average rank” across eps and compare to per-ε ranks

### 5.3 Visualizations
- [ ] Plot: robust accuracy vs ε (one line per model) → `figures/acc_vs_eps.png`
- [ ] Plot: rank position vs ε (optional) → `figures/rank_vs_eps.png`
- [ ] (optional) Heatmap: models × ε with robust accuracy

---

## 6) Write the report (short and clear)
Structure suggestion:
- [ ] **Setup**: models, dataset, subset size, ε sweep, AutoAttack version/config
- [ ] **Method**: how evaluation was run, device/batch size, reproducibility choices
- [ ] **Results**: table + plots + ranking changes
- [ ] **Discussion**:
  - [ ] Where rankings are stable vs unstable
  - [ ] Any surprising crossovers (model A > B at low ε but not at high ε)
  - [ ] Limitations: small subset, runtime constraints, randomness (if any)
- [ ] **Conclusion**: main takeaway on “ranking stability” across ε

---

## 7) Reproducibility checklist (don’t skip)
- [ ] Fix and log random seeds
- [ ] Save subset indices
- [ ] Log library versions + GPU/CPU info
- [ ] Commit config + scripts
- [ ] Make sure results can be regenerated with **one command**

---

## 8) Optional improvements
- [ ] Repeat with a different random subset (e.g., 3 runs) and show variability
- [ ] Add confidence intervals via bootstrapping on the subset
- [ ] Compare “standard” vs another AutoAttack version (if allowed)
- [ ] Extend ε grid slightly (within reason)

---

## Command checklist (example)
- [ ] `python src/run_autoattack_sweep.py --config config.yaml`
- [ ] `python src/analyze_rankings.py --in results/robust_accuracy.csv`
- [ ] Check `figures/` outputs and final tables
