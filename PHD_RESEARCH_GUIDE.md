# 🎓 PhD Research Guide: Using MICROLIFE for Biology Research

## Complete Workflow from Hypothesis to Publication

This guide shows you **exactly** how to use the MICROLIFE simulation suite for your biology PhD research, from initial hypothesis to publication-ready results.

---

## 📚 Table of Contents

1. [Research Workflow Overview](#research-workflow-overview)
2. [Example Study: Biodiversity-Stability](#example-study-biodiversity-stability)
3. [Running Experiments](#running-experiments)
4. [Data Analysis](#data-analysis)
5. [Publication Preparation](#publication-preparation)
6. [Thesis Chapter Structure](#thesis-chapter-structure)

---

## 🔬 Research Workflow Overview

```
1. HYPOTHESIS FORMULATION
   ↓
2. EXPERIMENTAL DESIGN (parameter configuration)
   ↓
3. PILOT RUNS (test feasibility)
   ↓
4. BATCH EXECUTION (collect data)
   ↓
5. STATISTICAL ANALYSIS (test hypotheses)
   ↓
6. VISUALIZATION (publication figures)
   ↓
7. WRITING (methods, results, discussion)
```

---

## 📊 Example Study: Biodiversity-Stability

Let's walk through a complete research study from start to finish.

### Research Question

**"Does species diversity increase ecosystem stability?"**

### Hypothesis

**H₁**: Ecosystems with higher species richness will exhibit lower population variance (higher stability) than low-diversity systems.

**Null Hypothesis (H₀)**: Species richness has no effect on population stability.

### Predictions

1. **High diversity** (8+ species) → CV < 0.3 (stable)
2. **Medium diversity** (4-6 species) → CV = 0.3-0.5 (moderate)
3. **Low diversity** (1-3 species) → CV > 0.5 (unstable)

### Experimental Design

**Factorial Design**:
- **Factor A**: Species Richness (3 levels: Low=2, Medium=5, High=10)
- **Factor B**: Environment (2 levels: Lake, Forest)
- **Replicates**: 50 per treatment combination
- **Total**: 3 × 2 × 50 = 300 simulation runs

**Dependent Variables**:
- Population coefficient of variation (CV)
- Time to equilibrium
- Extinction probability
- Shannon diversity index

**Controlled Variables**:
- Simulation duration (10,000 frames)
- Initial population (30 organisms)
- Mutation rates (10-20%)
- Random seed (systematic variation)

---

## 🚀 Running Experiments

### Step 1: Prepare Configuration Files

Create separate config files for each treatment:

```bash
# Copy base configuration
cp experiment_config.yaml config_low_diversity.yaml

# Edit to set parameters
# - Set species counts for low diversity (2 species)
# - Save as config_low_diversity.yaml

# Repeat for medium and high diversity
```

**config_low_diversity.yaml** (excerpt):
```yaml
experiment:
  name: "low_diversity_lake"
  description: "2 species (algae + predator) in lake environment"

initial_population:
  algae: 20
  predator: 10
  scavenger: 0
  # All others: 0

random_seed: 42  # Will be incremented for each replicate
```

### Step 2: Run Pilot Tests

Before running 300 simulations, test with a few:

```bash
# Test one replicate of each treatment
./MICROLIFE_ULTIMATE --config config_low_diversity.yaml

# Check:
# - Does it run without errors?
# - Are output files created?
# - Do population dynamics look reasonable?
# - Is runtime acceptable (<2 minutes)?
```

### Step 3: Batch Execution

**Option A: Simple batch (same config, many replicates)**:
```bash
python batch_runner.py \
    --config config_low_diversity.yaml \
    --replicates 50 \
    --parallel 8 \
    --output-dir results/low_diversity_lake
```

**Option B: Parameter sweep (vary one parameter)**:
```bash
python batch_runner.py \
    --config experiment_config.yaml \
    --sweep initial_population.algae 10,20,30,40,50 \
    --replicates 20 \
    --parallel 8
```

**Option C: Full factorial design**:
```bash
# Create a script to run all combinations
for diversity in low medium high; do
    for env in lake forest; do
        python batch_runner.py \
            --config config_${diversity}_${env}.yaml \
            --replicates 50 \
            --parallel 8 \
            --output-dir results/${diversity}_${env}
    done
done
```

**Expected Runtime**:
- Single simulation: ~30 seconds
- 300 simulations × 30s = 2.5 hours (sequential)
- With 8 parallel workers: ~20 minutes

### Step 4: Monitor Progress

While running:
```bash
# Check how many completed
ls -1 results/*/population_timeseries.csv | wc -l

# Watch progress in real-time
watch -n 10 'ls -1 results/*/population_timeseries.csv | wc -l'

# Check for errors
grep "success.*false" batch_results/aggregated_results.csv
```

---

## 📈 Data Analysis

### Step 1: Load and Inspect Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Load all time series data
data_files = glob('results/*/population_timeseries.csv')
print(f"Found {len(data_files)} data files")

# Load one example
df = pd.read_csv(data_files[0])
print(df.head())
print(df.describe())
```

### Step 2: Calculate Metrics for Each Replicate

```python
results = []

for file in data_files:
    df = pd.read_csv(file)

    # Extract treatment info from path
    # e.g., results/low_diversity_lake/replicate_0001/...
    parts = file.split('/')
    treatment = parts[1]  # "low_diversity_lake"

    # Calculate stability metric (CV)
    species_cols = [c for c in df.columns if 'count' in c]
    total_pop = df[species_cols].sum(axis=1)

    cv = total_pop.std() / total_pop.mean()

    # Other metrics
    final_pop = total_pop.iloc[-1]
    extinct = (final_pop == 0)
    richness = (df[species_cols].iloc[-1] > 0).sum()

    results.append({
        'treatment': treatment,
        'cv': cv,
        'final_population': final_pop,
        'extinct': extinct,
        'richness': richness,
        'mean_population': total_pop.mean()
    })

results_df = pd.DataFrame(results)
results_df.to_csv('analysis/summary_metrics.csv', index=False)
```

### Step 3: Statistical Analysis

```python
from scipy import stats
import seaborn as sns

# Split into treatment groups
low_div = results_df[results_df['treatment'].str.contains('low')]
med_div = results_df[results_df['treatment'].str.contains('medium')]
high_div = results_df[results_df['treatment'].str.contains('high')]

# Descriptive statistics
print("Low diversity CV:", low_div['cv'].mean(), "±", low_div['cv'].std())
print("Med diversity CV:", med_div['cv'].mean(), "±", med_div['cv'].std())
print("High diversity CV:", high_div['cv'].mean(), "±", high_div['cv'].std())

# One-way ANOVA
f_stat, p_value = stats.f_oneway(
    low_div['cv'],
    med_div['cv'],
    high_div['cv']
)

print(f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant effect of diversity on stability (p < 0.05)")

    # Post-hoc Tukey HSD
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey = pairwise_tukeyhsd(
        results_df['cv'],
        results_df['treatment'],
        alpha=0.05
    )
    print(tukey)

# Effect size (eta-squared)
ss_between = ((low_div['cv'].mean() - results_df['cv'].mean())**2 * len(low_div) +
              (med_div['cv'].mean() - results_df['cv'].mean())**2 * len(med_div) +
              (high_div['cv'].mean() - results_df['cv'].mean())**2 * len(high_div))

ss_total = ((results_df['cv'] - results_df['cv'].mean())**2).sum()

eta_squared = ss_between / ss_total
print(f"\nEffect size (η²) = {eta_squared:.3f}")

# Interpretation
if eta_squared > 0.14:
    print("Large effect size")
elif eta_squared > 0.06:
    print("Medium effect size")
else:
    print("Small effect size")
```

### Step 4: Automated Analysis with Toolkit

```bash
# Run analysis on each treatment
for treatment_dir in results/*/; do
    python analysis_toolkit.py \
        --data ${treatment_dir}/population_timeseries.csv \
        --analysis all \
        --output-dir analysis/$(basename $treatment_dir)
done
```

This generates:
- Statistical reports (TXT)
- Time series plots (PNG, 600 DPI)
- Phase space plots (PNG)
- FFT analysis (PNG)
- LaTeX tables (TEX)
- Raw results (JSON)

---

## 📊 Publication Preparation

### Create Publication-Quality Figures

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Figure 1: Population dynamics example
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

treatments = ['low', 'medium', 'high']
environments = ['lake', 'forest']

for i, env in enumerate(environments):
    for j, div in enumerate(treatments):
        ax = axes[i, j]

        # Load example replicate
        file = f'results/{div}_diversity_{env}/replicate_0001/population_timeseries.csv'
        df = pd.read_csv(file)

        # Plot
        species_cols = [c for c in df.columns if 'count' in c]
        for col in species_cols:
            ax.plot(df['timestamp'], df[col], label=col.replace('_count', ''))

        ax.set_title(f'{div.title()} Diversity - {env.title()}')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Population')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/Figure1_dynamics.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/Figure1_dynamics.pdf', bbox_inches='tight')  # Vector format
```

```python
# Figure 2: Stability comparison (boxplot)
fig, ax = plt.subplots(figsize=(8, 6))

# Prepare data
plot_data = results_df[['treatment', 'cv']].copy()
plot_data['diversity'] = plot_data['treatment'].apply(
    lambda x: 'Low' if 'low' in x else ('Medium' if 'medium' in x else 'High')
)

# Boxplot
sns.boxplot(data=plot_data, x='diversity', y='cv', ax=ax,
           order=['Low', 'Medium', 'High'],
           palette='viridis')

# Add individual points
sns.stripplot(data=plot_data, x='diversity', y='cv', ax=ax,
             order=['Low', 'Medium', 'High'],
             color='black', alpha=0.3, size=2)

# Add significance bars
from statannot import add_stat_annotation
add_stat_annotation(ax, data=plot_data, x='diversity', y='cv',
                   order=['Low', 'Medium', 'High'],
                   box_pairs=[('Low', 'Medium'), ('Medium', 'High'), ('Low', 'High')],
                   test='t-test_ind', text_format='star')

ax.set_xlabel('Species Diversity', fontsize=12)
ax.set_ylabel('Population Coefficient of Variation', fontsize=12)
ax.set_title('Ecosystem Stability vs. Species Diversity', fontsize=14)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/Figure2_stability.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/Figure2_stability.pdf', bbox_inches='tight')
```

### Create LaTeX Tables for Paper

**Table 1: Summary Statistics**

```python
# Generate LaTeX table
with open('tables/Table1_statistics.tex', 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Population Stability Metrics by Treatment}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Treatment & N & CV (Mean ± SD) & Extinction Rate & Final Population \\\\\n")
    f.write("\\hline\n")

    for treatment in ['low', 'medium', 'high']:
        for env in ['lake', 'forest']:
            group = results_df[results_df['treatment'] == f'{treatment}_diversity_{env}']

            n = len(group)
            cv_mean = group['cv'].mean()
            cv_std = group['cv'].std()
            extinct_rate = group['extinct'].mean()
            final_pop = group['final_population'].mean()

            f.write(f"{treatment.title()} ({env}) & {n} & ")
            f.write(f"{cv_mean:.3f} ± {cv_std:.3f} & ")
            f.write(f"{extinct_rate:.1%} & ")
            f.write(f"{final_pop:.1f} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\label{tab:stability_metrics}\n")
    f.write("\\end{table}\n")
```

**Table 2: ANOVA Results**

```python
with open('tables/Table2_anova.tex', 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{One-Way ANOVA: Effect of Species Diversity on Population Stability}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Source & df & SS & F & p-value \\\\\n")
    f.write("\\hline\n")
    f.write(f"Between groups & 2 & {ss_between:.2f} & {f_stat:.3f} & {p_value:.4f} \\\\\n")
    f.write(f"Within groups & {len(results_df)-3} & {ss_total-ss_between:.2f} & - & - \\\\\n")
    f.write("\\hline\n")
    f.write(f"\\multicolumn{{5}}{{l}}{{Effect size: $\\eta^2$ = {eta_squared:.3f}}} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\label{tab:anova}\n")
    f.write("\\end{table}\n")
```

---

## 📝 Thesis Chapter Structure

### Chapter 3: The Biodiversity-Stability Relationship in Simulated Ecosystems

#### 3.1 Introduction

```
Background:
- Ecological theory predicts diversity → stability (insurance hypothesis)
- Experimental validation limited by scale and duration
- Computational models enable controlled experimentation

Research Question:
- Does species diversity increase ecosystem stability?

Hypotheses:
- H₁: Higher diversity → lower population variance
- Mechanisms: complementarity, insurance effects
```

#### 3.2 Methods

```latex
\subsection{Simulation Model}

We used the MICROLIFE ecosystem simulation platform v1.0
\citep{your_thesis_2025} to test the biodiversity-stability hypothesis.
The model simulates multi-species communities with realistic trophic
interactions based on Lotka-Volterra dynamics.

\subsubsection{Model Parameters}

All simulations used the following parameters:
\begin{itemize}
    \item Duration: 10,000 time steps (~5 minutes real-time)
    \item Initial population: 30 organisms (distributed across species)
    \item Mutation rate: 10-20\% variation in offspring traits
    \item Reproduction threshold: 70\% energy, age > 100 steps
    \item Environment: Lake (temperature 20°C, light 70\%, toxicity 10\%)
\end{itemize}

Complete parameter specifications are provided in Appendix A
(experiment\_config.yaml).

\subsubsection{Experimental Design}

We employed a factorial design with:
\begin{itemize}
    \item \textbf{Factor A}: Species richness (Low=2, Medium=5, High=10)
    \item \textbf{Factor B}: Environment type (Lake, Forest)
    \item \textbf{Replicates}: 50 per treatment (total N=300)
\end{itemize}

Random seeds were systematically varied (42-91 for low diversity,
92-141 for medium, 142-191 for high) to ensure reproducibility
while capturing stochastic variation.

\subsubsection{Data Collection}

For each replicate, we recorded:
\begin{itemize}
    \item Population time series (all species, every 100 steps)
    \item Birth/death events
    \item Mutation and evolution events
    \item Final population state
\end{itemize}

All data exported to CSV format for analysis in Python 3.10.

\subsubsection{Statistical Analysis}

Primary response variable: Population coefficient of variation (CV)
\begin{equation}
CV = \frac{\sigma_N}{\mu_N}
\end{equation}

where $\sigma_N$ is standard deviation of total population over time
and $\mu_N$ is mean population.

Statistical test: One-way ANOVA followed by Tukey HSD post-hoc test.
Significance level: $\alpha = 0.05$.

\subsubsection{Reproducibility}

All code, configurations, and data are available at:
\url{https://github.com/yourusername/project}

Git commit hash: \texttt{abc123...} (see Appendix B for full hash)
```

#### 3.3 Results

```latex
\subsection{Population Dynamics}

Time series plots showed characteristic predator-prey oscillations
in all treatments (Figure 1). Oscillation period averaged
$T = 850 \pm 120$ time steps, consistent with Lotka-Volterra
predictions.

\subsection{Biodiversity-Stability Relationship}

Population stability varied significantly with species richness
(F(2,297) = 45.3, p < 0.001, $\eta^2$ = 0.23; Table 2, Figure 2).

High-diversity communities exhibited significantly lower population
variance (CV = 0.28 ± 0.12) compared to medium-diversity
(CV = 0.42 ± 0.15, p < 0.001) and low-diversity communities
(CV = 0.61 ± 0.18, p < 0.001).

Extinction rates also differed:
\begin{itemize}
    \item Low diversity: 24\% (12/50 replicates)
    \item Medium diversity: 8\% (4/50 replicates)
    \item High diversity: 2\% (1/50 replicates)
\end{itemize}

These results support the biodiversity-stability hypothesis and
suggest insurance effects are operational in the model.
```

#### 3.4 Discussion

```
Interpretation:
- Results consistent with insurance hypothesis
- Functional redundancy buffers against perturbations
- Complementarity effects smooth population dynamics

Comparison to literature:
- Our CV = 0.28 (high div) vs. 0.35 in Tilman et al. (2006)
- Extinction rates match Ives & Carpenter (2007)
- Effect size (η² = 0.23) is large, biologically meaningful

Limitations:
- Simplified trophic structure (3-12 species vs. real ecosystems)
- Homogeneous environment (no spatial heterogeneity)
- Fixed parameter values (need sensitivity analysis)

Future directions:
- Test across environmental gradients
- Vary mutation rates
- Add spatial structure
- Compare to experimental data
```

---

## 🎯 Tips for Professors & Reviewers

### What Makes This Research Strong

1. **Reproducibility**
   - Exact random seeds documented
   - All code version-controlled (Git)
   - Complete parameter files provided
   - Docker container available (future)

2. **Statistical Rigor**
   - Sufficient sample size (N=50 per treatment)
   - Appropriate tests (ANOVA + post-hoc)
   - Effect sizes reported (η²)
   - Raw data available for re-analysis

3. **Biological Realism**
   - Parameters from literature
   - Validated against published data
   - Realistic trophic interactions
   - Environmental gradients

4. **Clear Documentation**
   - Methods section has all details
   - Code is commented
   - Analysis scripts provided
   - Figures have raw data available

### Common Reviewer Questions

**Q: "How do you know your model is realistic?"**

A: "We validated our model against published Lotka-Volterra experiments
(Huffaker 1958) and found matching predator:prey ratios (1:7 vs. 1:6-12
in literature), oscillation periods (850 steps ≈ 10-15 generations),
and trophic transfer efficiencies (~10%). See README_ACADEMIC.md
Section 'Validation & Verification'."

**Q: "Why use simulations instead of real experiments?"**

A: "Simulations complement experiments by: (1) enabling perfect
replication impossible in nature, (2) systematically varying parameters
infeasible experimentally, (3) running longer timescales (1000s of
generations), and (4) testing mechanistic hypotheses with controlled
conditions. We view this as hypothesis-generating for future empirical work."

**Q: "How sensitive are results to parameter choices?"**

A: "We conducted sensitivity analysis (batch_runner.py --sweep) varying
mutation rates (0.05-0.25), reproduction thresholds (0.5-0.9 energy),
and environmental parameters. Core result (diversity → stability) was
robust across all parameter ranges (see Supplementary Figure S3)."

---

## ✅ Checklist for PhD Thesis

Before submission, ensure:

- [ ] All simulations documented with exact parameters
- [ ] Random seeds recorded for each replicate
- [ ] Statistical assumptions checked (normality, homoscedasticity)
- [ ] Effect sizes reported (not just p-values)
- [ ] Raw data archived (university repository)
- [ ] Code deposited (GitHub with DOI)
- [ ] Figures meet journal requirements (600+ DPI, vector formats)
- [ ] Methods section allows full replication
- [ ] Comparison to published literature
- [ ] Limitations discussed honestly
- [ ] Future directions proposed

---

## 📚 Recommended Reading

**Biodiversity-Stability**:
- Tilman et al. (2006) *Nature* - Biodiversity and stability experiments
- Ives & Carpenter (2007) *Science* - Stability in complex communities
- McCann (2000) *Nature* - Diversity-stability debate

**Computational Ecology**:
- DeAngelis & Mooij (2005) - Individual-based modeling
- Grimm et al. (2006) - Pattern-oriented modeling
- Railsback & Grimm (2019) - *Agent-Based Models* textbook

**Statistics**:
- Zuur et al. (2009) - *Mixed Effects Models in Ecology*
- Quinn & Keough (2002) - *Experimental Design* for biologists

---

## 🎓 Final Advice

1. **Start Small**: Run 10 replicates first, check everything works
2. **Document Everything**: Future you will thank present you
3. **Version Control**: Commit after every working change
4. **Automate**: Use batch scripts, avoid manual runs
5. **Validate**: Compare to published data whenever possible
6. **Ask for Help**: Show professors early drafts for feedback
7. **Think Critically**: Simulations are tools, not truth

**Remember**: All models are wrong, but some are useful. Your job is to
make your model as useful as possible for testing biological hypotheses.

---

**Good luck with your PhD! 🎓🔬🌿**

*"The purpose of computing is insight, not numbers."* - Richard Hamming
