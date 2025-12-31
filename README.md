# MLSECproject
Repo for Machine Learning Security exam project

## Project 1 — To‑Do Checklist (RobustBench + AutoAttack)

Goal: **Re-evaluate 5 RobustBench CIFAR-10 (L∞) models with AutoAttack** on a **100–200 image subset**, sweeping **ε**, and study how the **ranking** changes with ε.

---

## 0) Experiment spec 
- [x] **Pick 5 models** from RobustBench: dataset=`cifar10`, threat_model=`Linf`
- [x] Prefer a mix of strong + medium models (so ranking can change)
- [x] Choose **subset size**: `n_examples ∈ {100, 200} = 150`
- [x] Choose **ε sweep** = `[1, 4, 8, 12, 16] / 255`
- [x] Choose batch size `bs` = 50 
- [x] Fix random seed = 0 

Deliverables to produce:
- [ ] Table: robust accuracy for each model at each ε
- [ ] Plot(s): robust accuracy vs ε (lines per model)
- [ ] Ranking analysis: ranks per ε + rank correlation / rank flips
- [ ] Short write-up: methods, results, discussion, limitations

---

## 1) Environment setup
- [x] Install dependencies
  - [x] `torch` (CUDA build if available)
  - [x] `robustbench`
  - [x] `autoattack`
  - [x] `numpy`, `pandas`, `matplotlib`
  - [x] (optional) `scipy` for Spearman correlation

Example (pip):
- [x] `pip install torch torchvision torchaudio` (choose the right CUDA/CPU wheel)
- [x] `pip install robustbench numpy pandas matplotlib scipy`
- [x] install autoattack from its repo `pip install git+https://github.com/fra31/auto-attack`

---

## 2) Implement the evaluation script
Create a main script that:

### 2.1 Load data (RobustBench helper)
- [x] Load CIFAR-10 test set subset using RobustBench utilities
- [x] Move tensors to `device` (`cuda` if available)
- [x] Store the chosen indices (for reproducibility)
  - [x] Save indices to `results/subset_indices.npy`

### 2.2 Load 5 models (RobustBench)
- [x] For each model name:
  - [x] `load_model(model_name=..., dataset="cifar10", threat_model="Linf")`
  - [x] `model.eval()` and move to device
  - [x] (optional) run a clean accuracy check on the subset

### 2.3 Run AutoAttack for each ε
- [x] For each model and each ε:
  - [x] Instantiate `AutoAttack(model, norm="Linf", eps=eps, version="standard")`
  - [x] Run: `run_standard_evaluation(x, y, bs=bs)`
  - [x] Compute robust accuracy on the returned adversarial examples
  - [x] Save results in a structured format (dict → CSV/JSON)
  
--- 
## 3) Save results 
- [x] Save a **CSV**: rows = models, columns = eps, values = robust accuracy
- [x] Save a **JSON** with metadata:
  - [x] model names, eps list, subset size, batch size, seed
---


## 4) Analysis: ranking vs ε
Create an analysis notebook or script that:

### 4.1 Rankings per ε
- [ ] For each ε:
  - [ ] Rank models by robust accuracy (descending)
  - [ ] Save ranking table: `results/rankings.csv`

### 4.3 Visualizations
- [ ] Plot: robust accuracy vs ε (one line per model) → `plots/acc_vs_eps.png`
- [ ] Plot: rank position vs ε (optional) → `plots/rank_vs_eps.png`
- [ ] Plot: Heatmap: models × ε with robust accuracy
- [ ] Plot: robust accuracy vs ε with RobustBench computed accuracy → `plots/results_acc_vs_eps_with_rb.png`
- [ ] Plot: 8/255 robust accuracy vs RobustBench Robust accuracy
---


## 5) Write the report
Structure suggestion:
- [ ] **Setup**: models, dataset, subset size, ε sweep, AutoAttack version/config
- [ ] **Method**: how evaluation was run, device/batch size, reproducibility choices
- [ ] **Results**: table + plots + ranking changes
- [ ] **Discussion**:
  - [ ] Where rankings are stable vs unstable
  - [ ] Any surprising crossovers (model A > B at low ε but not at high ε)
  - [ ] Limitations: small subset, runtime constraints, randomness (if any)
- [ ] **Conclusion**: main takeaway on “ranking stability” across ε


