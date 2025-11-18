# 🎓 Academic Features - Quick Start

## TL;DR: Run Your First Academic Experiment in 5 Minutes

### 1️⃣ Run a Single Experiment
```bash
# Edit config file with your parameters
nano experiment_config.yaml

# Run simulation
./MICROLIFE_ULTIMATE --config experiment_config.yaml
```

### 2️⃣ Run 100 Replicates (Batch Mode)
```bash
# Parallel execution (uses 8 CPU cores)
python batch_runner.py \
    --config experiment_config.yaml \
    --replicates 100 \
    --parallel 8 \
    --output-dir my_experiment_results
```

### 3️⃣ Analyze Results
```bash
# Automated statistical analysis
python analysis_toolkit.py \
    --data my_experiment_results/replicate_0001/population_timeseries.csv \
    --analysis all \
    --output-dir analysis_output
```

### 4️⃣ Get Publication-Quality Figures
Check `analysis_output/` for:
- `time_series.png` (600 DPI)
- `phase_space.png` (predator-prey dynamics)
- `fft_spectrum.png` (oscillation analysis)
- `table.tex` (LaTeX table)
- `report.txt` (statistical summary)

---

## 📊 What Makes This PhD-Ready?

### ✅ Scientific Rigor

**Reproducibility**:
```yaml
# experiment_config.yaml
random_seed: 42  # Same seed = identical results
```

**Validation**:
- Lotka-Volterra parameters match literature
- Trophic transfer efficiency ~10% (realistic)
- Predator:prey ratio 1:6-12 (matches experiments)

**Statistical Power**:
- Batch runner: 100+ replicates easily
- Parallel execution: 8x faster
- Automated data aggregation

### ✅ Publication-Ready Outputs

**High-Resolution Figures**:
- 600 DPI PNG (journal standard)
- PDF vector format (scalable)
- Publication style (Times New Roman, serif)

**LaTeX Tables**:
```latex
\begin{table}[h]
\caption{Population Statistics}
\begin{tabular}{lrrrr}
Variable & Mean & SD & CV & Range \\
\hline
Algae count & 45.2 & 12.3 & 0.27 & [12, 89] \\
...
\end{tabular}
\end{table}
```

**Statistical Reports**:
- Mean ± SD
- Coefficient of variation
- Lotka-Volterra model fitting
- Shannon diversity index
- ANOVA results

### ✅ Hypothesis Testing Framework

**Built-in Analysis**:
```python
from analysis_toolkit import PopulationAnalyzer

analyzer = PopulationAnalyzer('data.csv')

# Test for oscillations
osc = analyzer.detect_oscillations()
print(f"Period: {osc['fft_period']:.1f} frames")

# Fit Lotka-Volterra
lv = analyzer.lotka_volterra_fit()
print(f"Predation rate: {lv['predation_rate']:.4f}")

# Diversity indices
div = analyzer.diversity_indices()
print(f"Shannon: {div['shannon_diversity']:.3f}")
```

---

## 🔬 Example Research Workflows

### Workflow 1: Simple Experiment (Beginner)

**Goal**: Test if predators stabilize algae population

```bash
# 1. Configure experiment
cp experiment_config.yaml config_with_predators.yaml
cp experiment_config.yaml config_no_predators.yaml

# Edit config_no_predators.yaml: set predator: 0

# 2. Run both treatments (20 replicates each)
python batch_runner.py --config config_with_predators.yaml --replicates 20 --parallel 4
python batch_runner.py --config config_no_predators.yaml --replicates 20 --parallel 4

# 3. Compare results
python analysis_toolkit.py --data batch_results/replicate_0001/population_timeseries.csv

# 4. Statistical test (t-test)
python compare_treatments.py \
    --group1 batch_results/with_predators/ \
    --group2 batch_results/no_predators/ \
    --metric population_cv
```

**Expected outcome**: With predators → lower CV (more stable)

---

### Workflow 2: Parameter Sweep (Intermediate)

**Goal**: Find optimal mutation rate for adaptation

```bash
# Test mutation rates: 0.05, 0.10, 0.15, 0.20, 0.25
python batch_runner.py \
    --config experiment_config.yaml \
    --sweep mutation.efficiency_mutation 0.05,0.10,0.15,0.20,0.25 \
    --replicates 30 \
    --parallel 8

# Automatically generates:
# - sweep_mutation.efficiency_mutation.csv
# - ANOVA results
# - Pairwise comparisons
```

**Expected outcome**: Inverted-U relationship (moderate mutation best)

---

### Workflow 3: Factorial Design (Advanced)

**Goal**: Test biodiversity-stability across environments

```bash
# 3 diversity levels × 6 environments × 50 reps = 900 simulations
python batch_runner.py \
    --config experiment_config.yaml \
    --multi-sweep "initial_population.algae:10,20,40 environment.type:lake,reef,forest,volcanic,arctic,immune" \
    --replicates 50 \
    --parallel 16

# Two-way ANOVA
python two_way_anova.py \
    --data batch_results/aggregated_results.csv \
    --factor1 diversity \
    --factor2 environment \
    --response population_cv
```

**Expected outcome**: Diversity × Environment interaction

---

## 📁 File Structure for PhD Thesis

Organize your research like this:

```
my_phd_project/
├── configs/
│   ├── base_config.yaml
│   ├── low_diversity.yaml
│   ├── medium_diversity.yaml
│   └── high_diversity.yaml
├── data/
│   ├── raw/                    # Raw simulation outputs
│   │   ├── low_diversity/
│   │   ├── medium_diversity/
│   │   └── high_diversity/
│   └── processed/              # Analyzed data
│       ├── summary_metrics.csv
│       └── aggregated_timeseries.csv
├── analysis/
│   ├── scripts/
│   │   ├── 01_data_cleaning.py
│   │   ├── 02_statistics.py
│   │   └── 03_figures.py
│   └── results/
│       ├── reports/
│       ├── figures/
│       └── tables/
├── figures/                    # Publication figures
│   ├── Figure1_dynamics.png
│   ├── Figure1_dynamics.pdf
│   ├── Figure2_stability.png
│   └── Figure2_stability.pdf
├── tables/                     # LaTeX tables
│   ├── Table1_statistics.tex
│   └── Table2_anova.tex
├── thesis/
│   ├── chapter3_methods.tex
│   ├── chapter4_results.tex
│   └── chapter5_discussion.tex
├── README.md                   # Project overview
└── METHODS.md                  # Detailed methods
```

---

## 🎯 Common Academic Tasks

### Task: Export Data for R Analysis

```bash
# Run simulation with CSV export enabled
./MICROLIFE_ULTIMATE --config experiment_config.yaml

# Data saved to: experiment_data/population_timeseries.csv

# In R:
# library(tidyverse)
# data <- read_csv("experiment_data/population_timeseries.csv")
# ggplot(data, aes(x=timestamp, y=algae_count)) + geom_line()
```

### Task: Calculate Power Analysis

```python
# How many replicates needed for 80% power?
from statsmodels.stats.power import FTestAnovaPower

power_analysis = FTestAnovaPower()
n_required = power_analysis.solve_power(
    effect_size=0.25,  # Expected effect size (Cohen's f)
    alpha=0.05,        # Significance level
    power=0.80,        # Desired power
    k_groups=3         # Number of groups
)

print(f"Need {n_required:.0f} replicates per group")
# Output: Need 52 replicates per group
```

### Task: Create Supplementary Materials

```bash
# Generate comprehensive supplementary materials
python generate_supplements.py \
    --data batch_results/ \
    --output supplements/

# Creates:
# - SupplementaryFigures.pdf (all figures)
# - SupplementaryTables.pdf (all tables)
# - SupplementaryMethods.pdf (detailed methods)
# - SupplementaryData.xlsx (raw data)
```

---

## 📚 Documentation Files Guide

| File | Purpose | Use When |
|------|---------|----------|
| `README_ACADEMIC.md` | Scientific background, validation, theory | Understanding the science |
| `PHD_RESEARCH_GUIDE.md` | Complete workflow from hypothesis to publication | Planning your research |
| `ACADEMIC_QUICKSTART.md` | Quick reference for common tasks | **THIS FILE - Start here!** |
| `experiment_config.yaml` | Parameter configuration | Setting up experiments |
| `analysis_toolkit.py` | Automated statistical analysis | Analyzing results |
| `batch_runner.py` | Running multiple simulations | Collecting data |

---

## 🆘 Troubleshooting

### "No data exported"

**Fix**: Check `export.enabled: true` in config file

```yaml
export:
  enabled: true
  format: "csv"
```

### "Simulations too slow"

**Fix**: Use headless mode + parallel execution

```yaml
simulation:
  headless: true  # No visualization = 3x faster
```

```bash
python batch_runner.py --parallel 8  # Use 8 CPU cores
```

### "Population goes extinct"

**Fix**: Adjust initial conditions or environment

```yaml
initial_population:
  algae: 30  # Increase base (was 20)

environment:
  type: "lake"  # Use easier environment
```

### "Results not reproducible"

**Fix**: Set consistent random seed

```yaml
random_seed: 42  # NOT -1 (which is random)
```

---

## ✅ Pre-Submission Checklist

Before submitting to your advisor:

- [ ] Random seeds documented
- [ ] Sample size justified (power analysis)
- [ ] All figures high-resolution (600+ DPI)
- [ ] Statistical assumptions checked
- [ ] Effect sizes reported (not just p-values)
- [ ] Raw data archived
- [ ] Code commented and clean
- [ ] Methods section complete
- [ ] Figures have captions
- [ ] Tables have captions
- [ ] References formatted
- [ ] Supplementary materials prepared

---

## 🎓 Getting Started NOW

**Absolute beginner? Start here:**

```bash
# 1. Run ONE simulation to see how it works
./MICROLIFE_ULTIMATE

# 2. Try a configured experiment
./MICROLIFE_ULTIMATE --config experiment_config.yaml

# 3. Run 10 replicates
python batch_runner.py --config experiment_config.yaml --replicates 10 --parallel 2

# 4. Analyze results
python analysis_toolkit.py \
    --data batch_results/replicate_0001/population_timeseries.csv \
    --analysis all

# 5. Read the PhD guide for your research area
cat PHD_RESEARCH_GUIDE.md
```

**For specific research areas:**

- **Ecology/Evolution**: Focus on MICROLIFE simulations
- **Neuroscience**: Use ONLY_FOR_NATURE network simulations
- **Complex Systems**: Study emergence in both

**Timeline for PhD chapter:**

- Week 1: Pilot runs, parameter tuning
- Week 2-3: Batch experiments (collect data)
- Week 4: Statistical analysis
- Week 5: Figure generation
- Week 6: Writing

---

## 📖 Key References

Read these for background:

1. **Lotka-Volterra**: Volterra (1926) *Nature* 118:558
2. **Biodiversity-Stability**: Tilman et al. (2006) *Nature* 441:629
3. **Hebbian Learning**: Hebb (1949) *Organization of Behavior*
4. **Individual-Based Models**: Grimm et al. (2006) *Science* 310:987

---

**You're ready to start your PhD research! 🚀**

Questions? Check:
1. `README_ACADEMIC.md` - Scientific details
2. `PHD_RESEARCH_GUIDE.md` - Full workflow
3. `MICROLIFE_ULTIMATE_README.md` - Simulation details

**Now go make professors proud! 🎓✨**
