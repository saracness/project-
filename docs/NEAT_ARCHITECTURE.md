# NEAT: NeuroEvolution of Augmating Topologies

## 🧬 Overview

**NEAT** is a genetic algorithm for evolving artificial neural networks. Unlike traditional neuroevolution that uses fixed topologies, NEAT evolves **both weights and structure** of networks through genetic algorithms.

**Publication:** Stanley, K. O., & Miikkulainen, R. (2002). *Evolving Neural Networks through Augmenting Topologies*. Evolutionary Computation, 10(2), 99-127.

## 🎯 Key Innovations

### 1. **Historical Markings (Innovation Numbers)**
- Each gene (connection) has unique innovation number
- Tracks when structural mutations occurred
- Enables meaningful crossover of different topologies
- Solves competing conventions problem

### 2. **Speciation (Protecting Innovation)**
- Groups similar genomes into species
- Protects structural innovations
- Prevents premature extinction of novel structures
- Explicit fitness sharing within species

### 3. **Incremental Growth (Complexification)**
- Starts with minimal structure
- Adds complexity only when beneficial
- Avoids bloat in networks
- Searches efficiently through topology space

## 🏗️ Architecture

### Core Components

```
NEATGenome
├── NodeGene (neurons)
│   ├── id (unique identifier)
│   ├── type (input/hidden/output)
│   └── activation (sigmoid/tanh/relu)
│
├── ConnectionGene (synapses)
│   ├── innovation (global innovation number)
│   ├── in_node / out_node
│   ├── weight (connection strength)
│   └── enabled (can be disabled by mutation)
│
└── Fitness (evaluated performance)

NEATPopulation
├── Species (groups of similar genomes)
│   ├── representative (species exemplar)
│   ├── members (list of genomes)
│   ├── avg_fitness
│   └── stagnation counter
│
├── InnovationDB (global innovation tracking)
│   └── innovation_number → (in, out, innovation_id)
│
└── Config (hyperparameters)
```

## 🔬 Genetic Operators

### Mutation

#### **1. Add Connection Mutation**
```
Before:        After:
  A              A
  |              |╲
  C              | C
                 |/
                 B
```
- Probability: 5-10%
- Creates new connection between existing nodes
- Assigns new innovation number if novel
- Weight initialized randomly

#### **2. Add Node Mutation**
```
Before:        After:
  A              A
  |              |
  |w             |1.0
  ↓              ↓
  B              N
                 |w
                 ↓
                 B
```
- Probability: 1-3%
- Splits existing connection
- Old connection disabled
- Two new connections created
- Input connection weight = 1.0
- Output connection weight = old weight

#### **3. Weight Mutation**
- Uniform perturbation: 80%
- Complete replacement: 20%
- Allows fine-tuning of behavior

### Crossover

```
Parent 1:  1--2--3--4
           └─────5

Parent 2:  1--2--3--4
              └──6--7

Matching genes: 1,2,3,4 (randomly chosen from parents)
Disjoint genes: 5 (from fitter parent)
Excess genes: 6,7 (from fitter parent)

Offspring: 1--2--3--4
           └─────5
              └──6--7
```

### Speciation

**Compatibility Distance:**

```
δ = (c1 * E) / N + (c2 * D) / N + c3 * W̄

Where:
  E = excess genes
  D = disjoint genes
  W̄ = average weight difference of matching genes
  N = number of genes in larger genome
  c1, c2, c3 = importance coefficients
```

**Species Assignment:**
- If δ < δ_threshold → same species
- Otherwise → new species or different species

## 📊 Fitness Evaluation

### Explicit Fitness Sharing

```
adjusted_fitness(i) = fitness(i) / Σ sh(δ(i,j))
                                    j∈species

sh(δ) = 1  if δ < δ_threshold
        0  otherwise
```

- Reduces fitness based on species size
- Protects niches
- Prevents one species from dominating

### Species Reproduction

1. **Eliminate stagnant species** (no improvement for N generations)
2. **Allocate offspring** proportional to adjusted fitness
3. **Reproduce within species**:
   - Top performers reproduce more
   - Bottom 20% don't reproduce
   - Champion copied without mutation

## 🎮 Implementation Details

### Hyperparameters

```python
NEAT_CONFIG = {
    # Population
    'population_size': 150,

    # Mutation rates
    'prob_add_connection': 0.05,
    'prob_add_node': 0.03,
    'prob_mutate_weight': 0.8,
    'prob_weight_replace': 0.1,
    'prob_toggle_enable': 0.01,

    # Speciation
    'compatibility_threshold': 3.0,
    'c1': 1.0,  # excess coefficient
    'c2': 1.0,  # disjoint coefficient
    'c3': 0.4,  # weight difference coefficient

    # Species
    'species_stagnation_threshold': 15,
    'elitism_threshold': 5,  # copy best if species > 5
    'survival_threshold': 0.2,  # top 20% reproduce

    # Network
    'initial_connections': 'fully_connected',
    'activation_functions': ['sigmoid', 'tanh', 'relu'],
}
```

### Innovation Database

```python
class InnovationDB:
    def __init__(self):
        self.innovations = {}  # (in, out) → innovation_id
        self.next_innovation = 1

    def get_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key in self.innovations:
            return self.innovations[key]
        else:
            innovation_id = self.next_innovation
            self.innovations[key] = innovation_id
            self.next_innovation += 1
            return innovation_id
```

## 📈 Evolution Metrics

### Tracking

1. **Population-level:**
   - Average fitness
   - Max fitness
   - Number of species
   - Average genome size

2. **Species-level:**
   - Species age
   - Best fitness
   - Average compatibility distance
   - Stagnation counter

3. **Genome-level:**
   - Number of nodes
   - Number of connections
   - Innovation count
   - Structural complexity

### Visualization

1. **Network Topology:**
   - Directed graph visualization
   - Node layers (input/hidden/output)
   - Connection weights (thickness/color)

2. **Phylogenetic Trees:**
   - Species lineage
   - Innovation events
   - Extinction events

3. **Fitness Landscapes:**
   - Generation vs fitness
   - Species diversity
   - Complexity over time

## 🔬 Scientific Analysis

### Statistical Tests

1. **Significance Testing:**
   - Mann-Whitney U test (comparing algorithms)
   - Wilcoxon signed-rank test (paired runs)

2. **Effect Size:**
   - Cohen's d
   - Success rate analysis

3. **Convergence Analysis:**
   - Plateau detection
   - Diversity metrics

### Publication-Quality Output

1. **LaTeX Tables:**
   - Performance comparison
   - Statistical significance
   - Hyperparameter sensitivity

2. **TikZ Figures:**
   - Network diagrams
   - Evolution trees
   - Performance graphs

## 🎯 Applications

### 1. Game AI
- Flappy Bird
- Pole balancing
- Mario playing
- Strategy games

### 2. Robot Control
- Bipedal walking
- Quadruped locomotion
- Arm manipulation

### 3. Function Approximation
- XOR problem
- Complex classification
- Regression tasks

### 4. Artificial Life
- **Our MicroLife Project!**
- Organism behavior evolution
- Survival strategies
- Emergent cooperation

## 📚 References

1. Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation*, 10(2), 99-127.

2. Stanley, K. O., & Miikkulainen, R. (2004). Competitive Coevolution through Evolutionary Complexification. *Journal of Artificial Intelligence Research*, 21, 63-100.

3. Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks. *Artificial Life*, 15(2), 185-212.

## 🚀 Implementation Plan

### Phase 1: Core NEAT (1-2 hours)
- [x] Genome representation
- [ ] Mutation operators
- [ ] Crossover
- [ ] Speciation
- [ ] Population management

### Phase 2: Evaluation (1 hour)
- [ ] Fitness evaluation
- [ ] MicroLife integration
- [ ] Batch evaluation

### Phase 3: Visualization (1 hour)
- [ ] Network topology plots
- [ ] Phylogenetic trees
- [ ] Evolution graphs

### Phase 4: Analysis & Paper (1 hour)
- [ ] Statistical analysis
- [ ] LaTeX paper generation
- [ ] Results compilation

---

**Status:** Implementation in progress
**Expected Completion:** 3-4 hours
**Quality Target:** Nature Journal submission-ready
