# Neuron Life Cycle Simulation: Scientific Documentation

## Overview

This documentation describes the biological basis for the neuron simulation system implemented in MicroLife. All aspects are based on peer-reviewed neuroscience literature.

---

## Table of Contents

1. [Evolutionary Origin of Neurons](#evolutionary-origin-of-neurons)
2. [Neuronal Life Cycle](#neuronal-life-cycle)
3. [Neuronal Morphology](#neuronal-morphology)
4. [Synaptic Plasticity](#synaptic-plasticity)
5. [Energy Metabolism](#energy-metabolism)
6. [Neurovascular Coupling](#neurovascular-coupling)
7. [Implementation Architecture](#implementation-architecture)
8. [References](#references)

---

## Evolutionary Origin of Neurons

### Timeline of Nervous System Evolution

Based on **Moroz (2009)** and **Ryan & Grant (2009)**:

#### Pre-neuronal Era (~800 Million Years Ago)
- **Sponges (Porifera)**: No neurons, chemical signaling via secreted factors
- **Cell-cell communication**: Gap junctions, diffusible messengers
- **Advantage**: Coordinated responses to environment without specialized cells

#### Early Nervous Systems (~600-580 Mya)

**Cnidarians (Jellyfish, Hydra)**:
- First true neurons
- **Nerve net organization**: Diffuse, non-centralized
- Symmetrical signaling (no head/tail)
- Simple reflexes (tentacle withdrawal, feeding)

**Ctenophores (Comb Jellies)**:
- Independent evolution of neurons? (Moroz, 2009)
- Distinct molecular toolkit from other animals
- Suggests neurons may have evolved multiple times

#### Bilaterian Revolution (~555 Mya)

**Key innovations**:
- **Centralized nervous systems**: Concentration of neurons into ganglia and brains
- **Cephalization**: Brain at anterior end
- **Segmentation**: Repeated neural structures
- **Sensory-motor integration**: Fast predator-prey interactions

#### Vertebrate Elaboration (~525 Mya - Present)

**Major developments**:
- **Neural crest cells**: Enable complex peripheral nervous system
- **Myelination**: 10-100× increase in conduction speed
- **Cortical expansion**: Mammalian neocortex for complex cognition
- **Hippocampal neurogenesis**: Adult brain plasticity (Gage, 2000)

### Environmental Requirements for Neuronal Evolution

1. **Multicellularity**: Coordinated cell behavior needed
2. **Sufficient oxygen**: Neurons are metabolically expensive (~20% of body's energy)
3. **Developmental patterning**: HOX genes, morphogen gradients
4. **Ecological pressure**: Predation favors fast sensory-motor circuits
5. **Chemical signaling evolution**: Co-option of ancient signaling molecules

### Key Molecular Innovations

| Innovation | Function | Evolutionary Origin |
|-----------|----------|-------------------|
| Voltage-gated Na⁺ channels | Action potential generation | ~600 Mya |
| Synaptic vesicle machinery | Neurotransmitter release | ~600 Mya |
| Neurotransmitter receptors | Signal reception | Pre-metazoan (ancient) |
| Myelin proteins | Fast axonal conduction | ~450 Mya (vertebrates) |
| Dendritic spines | Computational units | ~400 Mya |

---

## Neuronal Life Cycle

### Overview of Life Stages

Based on **Kempermann et al. (2015)** and **Yuan & Yankner (2000)**:

```
Neural Stem Cell → Neurogenesis → Migration → Differentiation →
Synaptogenesis → Mature Function → Apoptosis
```

---

### 1. Neurogenesis

**Definition**: Birth of new neurons from neural stem cells

**Key Paper**: Gage, F. H. (2000). *Mammalian neural stem cells.* Science, 287(5457), 1433-1438.

#### Where Neurogenesis Occurs

**In Embryonic Development**:
- Entire neural tube
- Radial glia serve as neural stem cells
- Peak production: ~250,000 neurons/minute in human fetal brain

**In Adult Mammals** (Kempermann et al., 2015):
1. **Subventricular zone (SVZ)**: Lateral ventricles
   - Produces olfactory bulb neurons
   - Thousands of new neurons daily
2. **Dentate gyrus (DG)** of hippocampus:
   - Produces granule cells
   - Hundreds of new neurons daily
   - Critical for memory formation

**Regulatory Factors**:
- **Promoting**: Exercise, enriched environment, learning, BDNF
- **Inhibiting**: Stress, aging, inflammation, sleep deprivation

#### Molecular Mechanisms

1. **Symmetric division**: Stem cell → 2 stem cells (self-renewal)
2. **Asymmetric division**: Stem cell → stem cell + neuroblast
3. **Neuroblast proliferation**: Transit amplifying cells
4. **Cell cycle exit**: Expression of proneural genes (NeuroD, Math1)

---

### 2. Migration

**Key Paper**: Hatten, M. E. (2002). *New directions in neuronal migration.* Science, 297(5587), 1660-1663.

#### Migration Modes

**Radial Migration**:
- Along radial glial fibers
- Cortical pyramidal neurons follow "inside-out" pattern
- Layer 6 neurons born first, layer 2/3 last

**Tangential Migration**:
- GABAergic interneurons from ganglionic eminences
- Migrate perpendicular to radial glia
- Guided by chemotactic factors (Slit/Robo, Semaphorins)

#### Guidance Cues

| Cue Type | Examples | Function |
|----------|----------|----------|
| Attractive | Netrin-1, BDNF | "Come this way" |
| Repulsive | Slit, Semaphorin-3A | "Avoid this area" |
| Adhesive | N-CAM, Integrins | Substrate attachment |
| Contact-mediated | Eph/Ephrin | Boundary formation |

---

### 3. Differentiation

**Key Paper**: Guillemot, F. (2007). *Cell fate specification in the mammalian telencephalon.* Progress in Neurobiology, 83(1), 37-52.

#### Neuron Type Specification

**Factors determining identity**:
1. **Birth location**: Cortical layer, brain region
2. **Birth timing**: Early vs late neurogenesis
3. **Transcription factors**: Tbr1, Ctip2, Satb2 (cortical layers)
4. **Morphogen gradients**: Sonic hedgehog, BMPs, Wnts

#### Morphological Differentiation

**Dendrite development** (Jan & Jan, 2010):
1. **Initial outgrowth**: Multiple neurites extend
2. **Dendrite selection**: Some neurites become dendrites
3. **Arborization**: Branching via cytoskeletal dynamics
4. **Spine formation**: Dendritic spines appear (Bonhoeffer & Yuste, 2002)

**Axon specification** (Barnes & Polleux, 2009):
1. **Breaking symmetry**: One neurite becomes axon
2. **Axon guidance**: Growth cone navigation
3. **Target recognition**: Synaptic partner matching
4. **Myelination**: Oligodendrocytes wrap axon (if applicable)

---

### 4. Synaptogenesis

**Key Paper**: Waites, C. L., Craig, A. M., & Garner, C. C. (2005). *Mechanisms of vertebrate synaptogenesis.* Annual Review of Neuroscience, 28, 251-274.

#### Synaptic Formation Steps

1. **Target recognition**: Axon finds correct postsynaptic partner
2. **Initial contact**: Filopodia interaction
3. **Adhesion**: Synaptic adhesion molecules (Neurexin-Neuroligin)
4. **Molecular assembly**:
   - **Presynaptic**: Synaptic vesicles, release machinery
   - **Postsynaptic**: Neurotransmitter receptors, scaffolding proteins
5. **Maturation**: Functional strengthening

#### Timeline

- **Peak synaptogenesis**: Postnatal weeks-months (species dependent)
- **Synaptic density**:
  - Human cortex peaks at ~age 2-3 years
  - ~15,000 synapses per cortical neuron at peak
  - Pruning reduces to ~7,000-10,000 in adults

---

### 5. Mature Function

#### Electrophysiology

**Resting State** (Hodgkin & Huxley, 1952):
- Membrane potential: -70 mV
- Na⁺/K⁺ ATPase maintains gradients
- Leak channels at equilibrium

**Action Potential Generation**:
1. **Depolarization**: Stimulus opens voltage-gated Na⁺ channels
2. **Rising phase**: Na⁺ influx → +40 mV
3. **Repolarization**: K⁺ efflux returns to negative
4. **Hyperpolarization**: Transient undershoot
5. **Refractory period**: 1-2 ms (Na⁺ channel inactivation)

**Synaptic Transmission** (Südhof, 2004):
1. Action potential invades axon terminal
2. Voltage-gated Ca²⁺ channels open
3. Ca²⁺ triggers vesicle fusion (SNARE proteins)
4. Neurotransmitter release into synaptic cleft
5. Postsynaptic receptor binding
6. Ion channel opening (AMPA, NMDA, GABA-A, etc.)

#### Computational Properties

**Dendritic Integration** (Spruston, 2008):
- **Linear summation**: Small inputs add linearly
- **Nonlinear integration**: NMDA spikes, dendritic spikes
- **Compartmentalization**: Independent dendritic branches
- **Coincidence detection**: Temporal synchrony matters

---

### 6. Apoptosis

**Key Paper**: Yuan, J., & Yankner, B. A. (2000). *Apoptosis in the nervous system.* Nature, 407(6805), 802-809.

#### Triggers for Neuronal Death

1. **Developmental pruning**:
   - ~50% of neurons die during development
   - Competition for neurotrophic factors
   - "Use it or lose it" principle

2. **Trophic factor withdrawal**:
   - Lack of NGF, BDNF, NT-3
   - Target-derived signals

3. **Synaptic inactivity**:
   - Unconnected neurons die
   - Activity-dependent survival

4. **Metabolic stress**:
   - Hypoxia, glucose deprivation
   - Mitochondrial failure

5. **Toxic insults**:
   - Excitotoxicity (excessive glutamate)
   - Oxidative stress
   - Protein aggregates (neurodegenerative disease)

#### Apoptotic Cascade

1. **Initiation**: Death signals activate pro-apoptotic proteins (Bax, Bad)
2. **Commitment**: Mitochondrial outer membrane permeabilization
3. **Execution**: Caspase activation (caspase-9 → caspase-3)
4. **DNA fragmentation**: CAD nuclease
5. **Cell shrinkage**: Cytoskeleton breakdown
6. **Phagocytosis**: Microglia clear debris

---

## Neuronal Morphology

### Classification Systems

**By morphology** (Ascoli et al., 2007 - NeuroMorpho.Org):
- **Pyramidal**: Triangular soma, apical dendrite
- **Stellate**: Star-shaped, radial dendrites
- **Granule**: Small soma, limited dendrites
- **Purkinje**: Elaborate planar dendritic tree

**By neurotransmitter**:
- **Glutamatergic**: Excitatory (~80% of cortical neurons)
- **GABAergic**: Inhibitory (~20% of cortical neurons)
- **Modulatory**: Dopamine, serotonin, acetylcholine, norepinephrine

**By projection**:
- **Projection neurons**: Long-range connections between brain areas
- **Interneurons**: Local connections within region

### Morphological Parameters

Based on **Spruston (2008)**:

#### Dendritic Properties

| Property | Range | Functional Impact |
|----------|-------|-------------------|
| Total length | 1,000-10,000 μm | Integration capacity |
| Branching complexity | Sholl intersections | Computational power |
| Spine density | 0-10 spines/μm | Synaptic input sites |
| Diameter | 0.5-5 μm | Electrical compartmentalization |

**Example - CA1 Pyramidal Neuron**:
- Basal dendrites: 3,000 μm total length
- Apical dendrite: 5,000 μm
- Spine count: 30,000 spines
- Synapses: ~30,000 inputs

#### Axonal Properties

| Property | Range | Functional Impact |
|----------|-------|-------------------|
| Length | 0.1 mm - 1 m | Projection range |
| Diameter | 0.2-20 μm | Conduction speed |
| Myelination | None to heavy | Speed (0.5 to 120 m/s) |
| Branching | 1-10,000 terminals | Divergence |

#### Soma Properties

- **Diameter**: 5-100 μm
- **Surface area**: Determines input resistance
- **Organelles**: Protein synthesis capacity

---

## Synaptic Plasticity

### Hebbian Learning

**Original principle** (Hebb, 1949):
> "Neurons that fire together, wire together"

**Modern formulation**: Correlated pre- and postsynaptic activity strengthens synapses

---

### Long-Term Potentiation (LTP)

**Key Paper**: Bliss, T. V., & Collingridge, G. L. (1993). *A synaptic model of memory: long-term potentiation in the hippocampus.* Nature, 361(6407), 31-39.

#### Discovery
- Bliss & Lømo (1973): High-frequency stimulation → lasting increase in synaptic strength
- Primarily studied in hippocampus
- Duration: Hours to days

#### Molecular Mechanisms

**Induction Phase** (0-60 seconds):
1. Glutamate release from presynaptic terminal
2. AMPA receptor activation → postsynaptic depolarization
3. NMDA receptor activation (requires depolarization + glutamate)
4. Ca²⁺ influx through NMDA receptors
5. CaMKII activation (calcium-calmodulin kinase II)

**Expression Phase** (minutes):
1. AMPA receptor phosphorylation → increased conductance
2. Additional AMPA receptors inserted into membrane
3. Dendritic spine enlargement

**Maintenance Phase** (hours-days):
1. Protein synthesis (Arc, CaMKII)
2. Structural changes: spine enlargement, stabilization
3. Presynaptic changes: increased neurotransmitter release

#### Types of LTP

- **Early LTP (E-LTP)**: 1-3 hours, protein synthesis-independent
- **Late LTP (L-LTP)**: >3 hours, requires protein synthesis and gene transcription

---

### Long-Term Depression (LTD)

**Key Paper**: Malenka, R. C., & Bear, M. F. (2004). *LTP and LTD: an embarrassment of riches.* Neuron, 44(1), 5-21.

#### Mechanisms

**Induction**:
- Low-frequency stimulation (1 Hz)
- Small Ca²⁺ increases (vs large for LTP)
- Phosphatase activation (PP1, calcineurin)

**Expression**:
- AMPA receptor dephosphorylation
- AMPA receptor endocytosis
- Spine shrinkage

**Function**:
- Synaptic weakening
- Forgetting/memory updating
- Homeostatic regulation

---

### Spike-Timing-Dependent Plasticity (STDP)

**Key Paper**: Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). *Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs.* Science, 275(5297), 213-215.

#### Rules

**Positive STDP** (LTP):
- Presynaptic spike **BEFORE** postsynaptic spike
- Δt = -20 to 0 ms
- Weight change: +20%

**Negative STDP** (LTD):
- Presynaptic spike **AFTER** postsynaptic spike
- Δt = 0 to +20 ms
- Weight change: -15%

#### Functional Significance
- Causality detection: "Pre caused post" → strengthen
- Temporal order encoding
- Sequence learning

---

### Structural Plasticity

**Key Paper**: Bonhoeffer, T., & Yuste, R. (2002). *Spine motility: phenomenology, mechanisms, and function.* Neuron, 35(6), 1019-1027.

#### Dendritic Spine Dynamics

**Spine types**:
- **Thin spines**: Small, mobile, transient (learning)
- **Mushroom spines**: Large, stable, strong (memory)
- **Stubby spines**: Short, intermediate

**Turnover rates**:
- Cortex: 5-10% spines turn over per month
- Hippocampus: Higher turnover, more plasticity
- Learning increases spine formation

**Activity-dependent changes**:
- LTP → spine enlargement (minutes-hours)
- LTD → spine shrinkage or elimination
- Stabilization requires repeated potentiation

---

## Energy Metabolism

### Overview

**Key Paper**: Magistretti, P. J., & Allaman, I. (2015). *A cellular perspective on brain energy metabolism and functional imaging.* Neuron, 86(4), 883-901.

#### Brain Energy Consumption

**Whole brain**:
- 20% of body's energy (only 2% of body mass)
- 20% of oxygen consumption
- 25% of glucose utilization
- Power: ~20 watts

**Per neuron**:
- 4.7 billion ATP molecules per second (cortical neuron)
- Glucose consumption: 5.5 × 10⁹ molecules/sec

---

### ATP Usage Breakdown

Based on **Attwell & Laughlin (2001)** - *An energy budget for signaling in the grey matter of the brain*:

| Process | % of Total Energy | Notes |
|---------|------------------|--------|
| Action potentials | 13% | Na⁺ influx, K⁺ efflux |
| Synaptic transmission | 50-60% | Vesicle cycling, receptors |
| Resting potential maintenance | 20-30% | Na⁺/K⁺ ATPase leak compensation |
| Housekeeping | ~10% | Protein synthesis, transport |

**Key insights**:
- Synapses are most expensive (neurotransmitter release, receptor trafficking)
- Dendrites consume more energy than soma or axon
- Action potentials are relatively "cheap" due to thin axons

---

### Glucose Metabolism Pathways

#### Glycolysis (Cytoplasm)
```
Glucose → 2 Pyruvate + 2 ATP + 2 NADH
```

#### Oxidative Phosphorylation (Mitochondria)
```
Pyruvate + O₂ → CO₂ + H₂O + ~30 ATP
```

**Total**: 1 glucose + 6 O₂ → ~32 ATP

**Efficiency**: ~40% (rest as heat)

---

### Astrocyte-Neuron Lactate Shuttle

**Key Paper**: Pellerin, L., & Magistretti, P. J. (1994). *Glutamate uptake into astrocytes stimulates aerobic glycolysis: a mechanism coupling neuronal activity to glucose utilization.* PNAS, 91(22), 10625-10629.

#### Mechanism

1. **Neuron fires** → Glutamate release
2. **Astrocyte takes up glutamate** → Na⁺ co-transport
3. **Na⁺/K⁺ ATPase activation** → Energy demand
4. **Astrocyte glycolysis** → Lactate production
5. **Lactate exported to neuron** → Converted to pyruvate
6. **Neuron uses pyruvate** → Oxidative metabolism

**Functional significance**:
- Astrocytes have glycogen stores (neurons don't)
- Lactate is efficient fuel during high activity
- Coupling mechanism for energy supply

---

## Neurovascular Coupling

### Overview

**Key Paper**: Attwell, D., Buchan, A. M., Charpak, S., Lauritzen, M., MacVicar, B. A., & Newman, E. A. (2010). *Glial and neuronal control of brain blood flow.* Nature, 468(7321), 232-243.

**Definition**: Mechanism linking neural activity to local blood flow increases

**Alternative names**: Functional hyperemia, hemodynamic response

---

### The Neurovascular Unit

**Components** (Iadecola, 2017):
1. **Neurons**: Activity generates metabolic demand
2. **Astrocytes**: Sense activity, signal to vessels
3. **Blood vessels**: Dilate/constrict to regulate flow
4. **Pericytes**: Contractile cells on capillaries (Hall et al., 2014)

**Integration**: These components function as unified system

---

### Vascular Anatomy

#### Blood Supply to Brain

**Arterial system**:
- Carotid arteries (anterior circulation)
- Vertebral arteries (posterior circulation)
- Circle of Willis (anastomotic ring)

**Microvascular system**:
- Arterioles: Resistance vessels, control flow
- Capillaries: Exchange vessels, gas/nutrient transfer
- Venules: Drainage

**Capillary density**:
- ~400 km of capillaries per 100g brain tissue
- Average intercapillary distance: 40 μm
- No neuron >25 μm from capillary

---

### Coupling Mechanisms

#### Signaling Pathways

**Vasodilatory signals** (increase blood flow):

1. **Nitric oxide (NO)**:
   - Source: Neuronal NOS, endothelial NOS
   - Mechanism: cGMP activation → smooth muscle relaxation
   - Timescale: Seconds

2. **Prostaglandin E2 (PGE2)**:
   - Source: Astrocytes (COX-2 pathway)
   - Mechanism: Smooth muscle relaxation
   - Timescale: Seconds to minutes

3. **Epoxyeicosatrienoic acids (EETs)**:
   - Source: Astrocytes
   - Mechanism: K⁺ channel activation
   - Timescale: Seconds

4. **Potassium (K⁺)**:
   - Source: Neurons (activity-dependent release)
   - Mechanism: Inward rectifying K⁺ channels
   - Timescale: Milliseconds to seconds

**Vasoconstrictive signals** (decrease blood flow):
- 20-HETE (arachidonic acid metabolite)
- Neuropeptide Y
- Adenosine (in some contexts)

---

### Temporal Dynamics

**Hemodynamic Response Function**:

```
Neural Activity Onset
↓
0-2 seconds: Initial dip (some studies)
↓
2-5 seconds: Rapid rise
↓
5-8 seconds: Peak response (~20-30% flow increase)
↓
8-20 seconds: Plateau
↓
20-30 seconds: Return to baseline
↓
30-60 seconds: Post-stimulus undershoot
```

**Spatial extent**:
- Localized to ~0.5-2 mm around active neurons
- Matches functional columns in cortex

---

### Functional Significance

#### Why Coupling Exists

1. **Energy delivery**: Neurons have high metabolic demands
2. **No energy storage**: Brain has ~1-2 minutes of stored ATP/glucose
3. **Oxygen dependence**: Neurons are obligate aerobic (minimal anaerobic capacity)
4. **Activity prediction**: Anticipatory regulation

#### Clinical Importance

**Functional neuroimaging**:
- fMRI (functional MRI): Detects BOLD signal (blood oxygen level-dependent)
- PET: Measures glucose uptake
- Both rely on neurovascular coupling

**Disease states**:
- **Alzheimer's**: Coupling dysfunction precedes neurodegeneration (Zlokovic, 2011)
- **Stroke**: Ischemia → oxygen/glucose deprivation
- **Hypertension**: Impaired autoregulation
- **Vascular dementia**: Chronic hypoperfusion

---

### Blood-Brain Barrier (BBB)

**Key Paper**: Abbott, N. J., Rönnbäck, L., & Hansson, E. (2006). *Astrocyte–endothelial interactions at the blood–brain barrier.* Nature Reviews Neuroscience, 7(1), 41-53.

#### Structure

**Components**:
1. **Endothelial cells**: Tight junctions (claudins, occludin)
2. **Basement membrane**: Extracellular matrix
3. **Pericytes**: Embedded in basement membrane
4. **Astrocyte endfeet**: Cover ~99% of capillary surface

#### Function

**Selective permeability**:
- **Lipid-soluble molecules**: O₂, CO₂, alcohol → free passage
- **Glucose**: GLUT1 transporter
- **Amino acids**: L-amino acid transporter
- **Large molecules**: Generally blocked
- **Ions**: Restricted (maintains brain homeostasis)

**Protection**:
- Excludes toxins, pathogens
- Efflux pumps (P-glycoprotein)
- Immune privilege

---

## Implementation Architecture

### File Structure

```
microlife/simulation/
├── neuron_morphology.py    # Morphological properties (dendrites, axons, spines)
├── neuron.py               # Neuron life cycle (neurogenesis → apoptosis)
└── neural_environment.py   # Tissue simulation (neurons + vessels + astrocytes)
```

---

### Class Hierarchy

#### NeuronMorphology Class
- **Physical traits**: Dendritic complexity, spine density, axon length, myelination
- **Ion channels**: Nav, Kv, Cav densities
- **Neurotransmitter**: Type and polarity (excitatory/inhibitory)
- **Derived properties**: Signal speed, integration capacity, metabolic demand
- **Methods**: `mutate()`, `get_description()`

#### Neuron Class
- **Position**: 3D coordinates
- **Life stage**: neurogenesis, migration, differentiation, mature, apoptotic
- **Energy state**: ATP level, glucose, oxygen
- **Electrophysiology**: Membrane potential, firing rate
- **Synaptic connectivity**: Incoming and outgoing synapses
- **Neurotrophic factors**: BDNF, NGF levels
- **Methods**:
  - `update()`: Main simulation loop
  - `_fire_action_potential()`: Spike generation
  - `_update_synaptic_plasticity()`: Hebbian learning
  - `_update_metabolism()`: Energy dynamics
  - `_initiate_apoptosis()`: Cell death

#### Synapse Class
- **Weight**: Synaptic efficacy (0-1)
- **Spine size**: Structural correlate of strength
- **Plasticity history**: LTP/LTD events
- **Stability**: Transient vs stable
- **Methods**: `apply_hebbian_plasticity()`

#### BloodVessel Class
- **Position**: 3D coordinates
- **Diameter**: Capillary size (3-7 μm)
- **Flow rate**: Current blood flow
- **Supply capacity**: Oxygen and glucose delivery
- **Dilation state**: Neurovascular coupling response
- **Methods**: `update()` - respond to neural activity

#### Astrocyte Class
- **Position**: 3D coordinates
- **Domain radius**: Territorial extent (30-60 μm)
- **Glycogen stores**: Energy reserves
- **Lactate production**: Astrocyte-neuron lactate shuttle
- **Ca²⁺ signaling**: Activity sensing
- **Vasodilator release**: Neurovascular coupling mediation
- **Methods**: `update()` - respond to neural activity

#### NeuralEnvironment Class
- **Components**: Lists of neurons, vessels, astrocytes
- **Dimensions**: 3D tissue volume
- **Environmental parameters**: O₂, glucose, temperature, neurotrophic factors
- **Methods**:
  - `initialize_vasculature()`: Create capillary network
  - `initialize_astrocytes()`: Create glial network
  - `create_neuron_at()`: Neurogenesis
  - `form_random_synapses()`: Synaptogenesis
  - `update()`: Main simulation step
  - `_update_vasculature()`: Neurovascular coupling
  - `_supply_metabolites_to_neurons()`: O₂/glucose delivery
  - `get_statistics()`: Simulation metrics

---

### Key Simulation Parameters

#### Neuron Parameters
- **Energy**: 0-200 ATP units
- **Membrane potential**: -70 mV (rest) to +40 mV (spike)
- **Firing threshold**: -55 mV (typical)
- **Refractory period**: 2 ms
- **Metabolic consumption**: 0.1-1.0 units/timestep

#### Plasticity Parameters
- **Learning rate**: 0.01 (default)
- **Synaptic weight range**: 0.0-1.0
- **Spine size range**: 0.1-1.0
- **Pruning threshold**: weight < 0.05

#### Environmental Parameters
- **Capillary spacing**: ~50 μm
- **Astrocyte domain**: 50 μm radius
- **Oxygen diffusion range**: <50 μm
- **Glucose diffusion range**: <50 μm
- **Neurovascular coupling delay**: 1-2 seconds
- **Max flow increase**: 30%

#### Timescales
- **Simulation timestep**: 0.1 seconds
- **Neurogenesis**: 10 time units
- **Migration**: 10 time units
- **Differentiation**: 10 time units
- **Synaptogenesis**: Ongoing during maturation

---

### Biological Fidelity vs Computational Efficiency

**Simplifications made**:
1. **Continuous time**: Real neurons fire discrete spikes
2. **Spatial abstraction**: No morphological compartments (soma, dendrites, axon are not geometrically represented)
3. **Simplified electrophysiology**: No Hodgkin-Huxley equations
4. **Population-level connectivity**: Not anatomically precise wiring
5. **Coarse-grained metabolism**: Simplified ATP dynamics

**Preserved biological principles**:
1. **Life cycle stages**: All major stages represented
2. **Energy constraints**: Realistic metabolic demands
3. **Neurovascular coupling**: Activity-flow relationship
4. **Synaptic plasticity**: Hebbian learning, LTP/LTD
5. **Evolutionary context**: Based on actual neural evolution
6. **Morphological diversity**: Real neuron types

---

## Usage Examples

### Create a Neural Environment

```python
from microlife.simulation.neural_environment import NeuralEnvironment

# Create tissue volume (500×500×100 μm)
env = NeuralEnvironment(width=500, height=500, depth=100)

# Initialize blood vessels (1% of volume)
env.initialize_vasculature(vessel_density=0.01)

# Initialize astrocytes
env.initialize_astrocytes(astrocyte_density=0.0001)

# Create neuronal population (100 neurons)
env.create_random_population(
    num_neurons=100,
    neuron_type_distribution={
        "pyramidal": 0.7,
        "granule": 0.1,
        "interneuron": 0.2
    }
)

# Form synaptic connections
env.form_random_synapses(connection_probability=0.05)

# Run simulation
for step in range(1000):
    env.update()

    if step % 100 == 0:
        env.print_statistics()
```

### Create Individual Neuron

```python
from microlife.simulation.neuron import Neuron
from microlife.simulation.neuron_morphology import create_neuron_morphology

# Create pyramidal neuron
morphology = create_neuron_morphology("pyramidal")
neuron = Neuron(x=100, y=100, z=50, morphology=morphology)

# Update over time
for t in range(100):
    neuron.update(dt=0.1, time=t*0.1)

print(neuron.get_state_description())
```

### Observe Synaptic Plasticity

```python
# Create two connected neurons
pre_neuron = Neuron(x=0, y=0, morphology=create_neuron_morphology("pyramidal"))
post_neuron = Neuron(x=50, y=0, morphology=create_neuron_morphology("pyramidal"))

# Form synapse
pre_neuron.connect_to_neuron(post_neuron, initial_weight=0.5)

# Simulate correlated activity (Hebbian learning)
for t in range(100):
    # Both active → LTP expected
    pre_neuron._fire_action_potential(t)
    post_neuron._fire_action_potential(t + 0.01)  # Just after pre

    pre_neuron.update(dt=0.1, time=t)
    post_neuron.update(dt=0.1, time=t)

# Check synaptic weight (should increase)
synapse = post_neuron.synapses_in[pre_neuron.id]
print(f"Final synaptic weight: {synapse.weight}")  # > 0.5
```

---

## Future Extensions

### Potential Additions

1. **Compartmental modeling**: Multi-compartment neurons (soma, dendrites, axon)
2. **Detailed ion channel dynamics**: Hodgkin-Huxley formalism
3. **Neuromodulation**: Dopamine, serotonin, acetylcholine effects
4. **Network oscillations**: Gamma, theta, alpha rhythms
5. **Pathology simulation**: Alzheimer's, Parkinson's, epilepsy
6. **Development**: Detailed embryonic patterning
7. **Learning tasks**: Memory consolidation, motor learning

### Research Questions

1. How does neurovascular coupling efficiency affect learning?
2. What connectivity patterns emerge from activity-dependent plasticity?
3. How does astrocyte density affect network stability?
4. What metabolic constraints limit network size?
5. Can evolutionary optimization discover efficient brain architectures?

---

## References

### Neuronal Life Cycle

1. **Gage, F. H. (2000)**. Mammalian neural stem cells. *Science*, 287(5457), 1433-1438.

2. **Kempermann, G., Song, H., & Gage, F. H. (2015)**. Neurogenesis in the adult hippocampus. *Cold Spring Harbor Perspectives in Biology*, 7(9), a018812.

3. **Hatten, M. E. (2002)**. New directions in neuronal migration. *Science*, 297(5587), 1660-1663.

4. **Guillemot, F. (2007)**. Cell fate specification in the mammalian telencephalon. *Progress in Neurobiology*, 83(1), 37-52.

5. **Yuan, J., & Yankner, B. A. (2000)**. Apoptosis in the nervous system. *Nature*, 407(6805), 802-809.

### Neuronal Morphology

6. **Spruston, N. (2008)**. Pyramidal neurons: dendritic structure and synaptic integration. *Nature Reviews Neuroscience*, 9(3), 206-221.

7. **Ascoli, G. A., Donohue, D. E., & Halavi, M. (2007)**. NeuroMorpho.Org: a central resource for neuronal morphologies. *Journal of Neuroscience*, 27(35), 9247-9251.

8. **Jan, Y. N., & Jan, L. Y. (2010)**. Branching out: mechanisms of dendritic arborization. *Nature Reviews Neuroscience*, 11(5), 316-328.

9. **Bonhoeffer, T., & Yuste, R. (2002)**. Spine motility: phenomenology, mechanisms, and function. *Neuron*, 35(6), 1019-1027.

10. **Luebke, J. I. (2017)**. Pyramidal neurons are not generalizable building blocks of cortical networks. *Frontiers in Neuroanatomy*, 11, 11.

### Synaptic Plasticity

11. **Hebb, D. O. (1949)**. *The Organization of Behavior: A Neuropsychological Theory*. Wiley, New York.

12. **Bliss, T. V., & Collingridge, G. L. (1993)**. A synaptic model of memory: long-term potentiation in the hippocampus. *Nature*, 361(6407), 31-39.

13. **Malenka, R. C., & Bear, M. F. (2004)**. LTP and LTD: an embarrassment of riches. *Neuron*, 44(1), 5-21.

14. **Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997)**. Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*, 275(5297), 213-215.

15. **Waites, C. L., Craig, A. M., & Garner, C. C. (2005)**. Mechanisms of vertebrate synaptogenesis. *Annual Review of Neuroscience*, 28, 251-274.

### Energy Metabolism

16. **Magistretti, P. J., & Allaman, I. (2015)**. A cellular perspective on brain energy metabolism and functional imaging. *Neuron*, 86(4), 883-901.

17. **Attwell, D., & Laughlin, S. B. (2001)**. An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133-1145.

18. **Pellerin, L., & Magistretti, P. J. (1994)**. Glutamate uptake into astrocytes stimulates aerobic glycolysis: a mechanism coupling neuronal activity to glucose utilization. *PNAS*, 91(22), 10625-10629.

### Neurovascular Coupling

19. **Attwell, D., Buchan, A. M., Charpak, S., Lauritzen, M., MacVicar, B. A., & Newman, E. A. (2010)**. Glial and neuronal control of brain blood flow. *Nature*, 468(7321), 232-243.

20. **Iadecola, C. (2017)**. The neurovascular unit coming of age: a journey through neurovascular coupling in health and disease. *Neuron*, 96(1), 17-42.

21. **Hall, C. N., Reynell, C., Gesslein, B., Hamilton, N. B., Mishra, A., Sutherland, B. A., ... & Attwell, D. (2014)**. Capillary pericytes regulate cerebral blood flow in health and disease. *Nature*, 508(7494), 55-60.

22. **Girouard, H., & Iadecola, C. (2006)**. Neurovascular coupling in the normal brain and in hypertension, stroke, and Alzheimer disease. *Journal of Applied Physiology*, 100(1), 328-335.

23. **Zlokovic, B. V. (2011)**. Neurovascular pathways to neurodegeneration in Alzheimer's disease and other disorders. *Nature Reviews Neuroscience*, 12(12), 723-738.

24. **Howarth, C. (2014)**. The contribution of astrocytes to the regulation of cerebral blood flow. *Frontiers in Neuroscience*, 8, 103.

25. **Abbott, N. J., Rönnbäck, L., & Hansson, E. (2006)**. Astrocyte–endothelial interactions at the blood–brain barrier. *Nature Reviews Neuroscience*, 7(1), 41-53.

### Evolution of Nervous Systems

26. **Moroz, L. L. (2009)**. On the independent origins of complex brains and neurons. *Brain, Behavior and Evolution*, 74(3), 177-190.

27. **Ryan, J. F., & Grant, K. A. (2009)**. Evolution of neural systems in Cnidaria. *Integrative and Comparative Biology*, 49(2), 142-153.

### Additional References

28. **Hodgkin, A. L., & Huxley, A. F. (1952)**. A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*, 117(4), 500-544.

29. **Südhof, T. C. (2004)**. The synaptic vesicle cycle. *Annual Review of Neuroscience*, 27, 509-547.

30. **Barnes, A. P., & Polleux, F. (2009)**. Establishment of axon-dendrite polarity in developing neurons. *Annual Review of Neuroscience*, 32, 347-381.

31. **Laughlin, S. B., & Sejnowski, T. J. (2003)**. Communication in neuronal networks. *Science*, 301(5641), 1870-1874.

32. **Tavosanis, G. (2012)**. Dendritic structural plasticity. *Developmental Neurobiology*, 72(1), 73-86.

---

## Conclusion

This neuron simulation system represents a biologically-grounded approach to modeling neural tissue. By incorporating principles from developmental neuroscience, electrophysiology, metabolism, and neurovascular coupling, it provides a platform for exploring:

- **Evolutionary constraints** on brain architecture
- **Metabolic trade-offs** in neural computation
- **Activity-dependent plasticity** and learning
- **Neurovascular coupling** dynamics
- **Life cycle** from birth to death

All components are based on peer-reviewed literature, making this not just a simulation, but an **interactive scientific model** of real biological processes.

---

*Document created: 2025*
*Last updated: 2025*
*Version: 1.0*
