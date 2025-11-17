# 🧠 Yapay Zeka Beyinleri - AI Brains Guide

## Mikroorganizmalara Farklı AI Modelleri ile Zeka Verildi!

Her grup farklı bir yapay zeka algoritması kullanarak hayatta kalmaya çalışıyor!

---

## 🎯 Implemented AI Models

### 1. **Q-Learning Brain** (Reinforcement Learning)
**Dosya:** `microlife/ml/brain_rl.py` → `QLearningBrain`

**Nasıl Çalışır:**
- Table-based reinforcement learning
- State'leri discretize eder (sürekli → ayrık)
- Q-table ile en iyi aksiyonu öğrenir
- Epsilon-greedy exploration

**Kullanım:**
```python
from microlife.ml.brain_rl import QLearningBrain

brain = QLearningBrain(learning_rate=0.1, epsilon=0.3)
action = brain.decide_action(state)
brain.learn(state, action, reward, next_state, done)
```

**Avantaj:** Basit, anlaşılır, garanti convergence
**Dezavantaj:** Büyük state space'lerde yavaş

---

### 2. **DQN Brain** (Deep Q-Network)
**Dosya:** `microlife/ml/brain_rl.py` → `DQNBrain`

**Nasıl Çalışır:**
- Neural network ile Q-values tahmin eder
- Experience replay ile öğrenir
- Continuous state space'ler için iyi

**Kullanım:**
```python
from microlife.ml.brain_rl import DQNBrain

brain = DQNBrain(state_size=7, hidden_size=32)
action = brain.decide_action(state)
brain.learn(state, action, reward, next_state, done)
```

**Avantaj:** Scalable, complex patterns
**Dezavantaj:** Training süresi uzun

---

### 3. **Double DQN Brain** (Modern RL)
**Dosya:** `microlife/ml/brain_rl.py` → `DoubleDQNBrain`

**Nasıl Çalışır:**
- DQN + Target network
- Overestimation bias'ı azaltır
- State-of-the-art RL tekniği

**Kullanım:**
```python
from microlife.ml.brain_rl import DoubleDQNBrain

brain = DoubleDQNBrain()
```

**Avantaj:** En stable RL yaklaşımı
**Dezavantaj:** Memory intensive

---

### 4. **CNN Brain** (Convolutional Neural Network)
**Dosya:** `microlife/ml/brain_cnn.py` → `CNNBrain`

**Nasıl Çalışır:**
- Çevreyi 2D grid olarak görür (visual perception!)
- Convolution layers ile pattern detection
- Biyolojik görsel korteksten esinlenmiş

**Kullanım:**
```python
from microlife.ml.brain_cnn import CNNBrain

brain = CNNBrain(grid_size=20)
action = brain.decide_action(state)
```

**Özellik:** Organizmanın "gördüğü" 20x20 grid:
- Food → 1.0
- Obstacles → -0.5
- Temperature zones → -0.5

**Avantaj:** Spatial patterns, vision-like
**Dezavantaj:** Computationally expensive

---

### 5. **ResNet-CNN Brain** (Residual Networks)
**Dosya:** `microlife/ml/brain_cnn.py` → `ResidualCNNBrain`

**Nasıl Çalışır:**
- CNN + skip connections
- Modern computer vision'dan
- Daha derin öğrenme

**Kullanım:**
```python
from microlife.ml.brain_cnn import ResidualCNNBrain

brain = ResidualCNNBrain(grid_size=20)
```

**Avantaj:** Deeper learning, better gradients

---

### 6. **Genetic Algorithm Brain** (Evolution)
**Dosya:** `microlife/ml/brain_evolutionary.py` → `GeneticAlgorithmBrain`

**Nasıl Çalışır:**
- Genome (20 gene) davranışı kodlar
- Mutation ile değişir
- Crossover ile çocuk üretir
- Doğal seleksiyon gibi!

**Genom Yapısı:**
- Gene 0-7: Yön tercihleri
- Gene 8-11: Enerji bazlı davranış
- Gene 12-15: Yemek arama
- Gene 16-19: Risk alma

**Kullanım:**
```python
from microlife.ml.brain_evolutionary import GeneticAlgorithmBrain

brain = GeneticAlgorithmBrain(genome_size=20, mutation_rate=0.1)
action = brain.decide_action(state)

# Evolution
brain.mutate()
child = brain1.crossover(brain2)
```

**Avantaj:** Biyolojik, evrimsel, explainable
**Dezavantaj:** Yavaş convergence

---

### 7. **NEAT Brain** (NeuroEvolution)
**Dosya:** `microlife/ml/brain_evolutionary.py` → `NEATBrain`

**Nasıl Çalışır:**
- Network STRUCTURE'ı da evolve eder!
- Başlangıçta minimal network
- Mutation: node ekle, connection ekle
- Weight evolution + topology evolution

**Kullanım:**
```python
from microlife.ml.brain_evolutionary import NEATBrain

brain = NEATBrain(input_size=7, output_size=9)
action = brain.decide_action(state)

# Evolve structure
brain.mutate(add_node_prob=0.03, add_conn_prob=0.05)
```

**Avantaj:** Discovers optimal architecture
**Dezavantaj:** Complex, many hyperparameters

---

### 8. **CMA-ES Brain** (Evolution Strategy)
**Dosya:** `microlife/ml/brain_evolutionary.py` → `CMAESBrain`

**Nasıl Çalışır:**
- Covariance Matrix Adaptation
- Evolution strategy (modern!)
- Distribution-based optimization
- Biyoloji araştırmalarında çok kullanılır

**Kullanım:**
```python
from microlife.ml.brain_evolutionary import CMAESBrain

brain = CMAESBrain(param_size=20)
action = brain.decide_action(state)

# Update distribution
brain.update_distribution(successful_params, fitness_values)
```

**Avantaj:** State-of-the-art evolution
**Dezavantaj:** Population-based (needs many organisms)

---

## 📊 Model Comparison Table

| Model | Type | Learning | Speed | Memory | Best For |
|-------|------|----------|-------|--------|----------|
| Q-Learning | RL | Online | Fast | Low | Simple envs |
| DQN | Deep RL | Batch | Medium | High | Complex states |
| Double DQN | Deep RL | Batch | Medium | High | Stable learning |
| CNN | Visual | Batch | Slow | High | Spatial tasks |
| ResNet-CNN | Visual | Batch | Slow | Very High | Deep vision |
| Genetic Alg | Evolution | Generational | Slow | Low | Interpretable |
| NEAT | Neuroevolution | Generational | Medium | Medium | Architecture search |
| CMA-ES | Evolution Strategy | Generational | Medium | Medium | Parameter tuning |

---

## 🎮 How to Use Different Brains

### Organizmalara Brain Atama:

```python
from microlife.simulation.organism import Organism
from microlife.ml.brain_rl import QLearningBrain, DQNBrain
from microlife.ml.brain_cnn import CNNBrain
from microlife.ml.brain_evolutionary import GeneticAlgorithmBrain, NEATBrain

# Create organisms with different brains
org1 = Organism(100, 100)
org1.brain = QLearningBrain()

org2 = Organism(200, 200)
org2.brain = DQNBrain()

org3 = Organism(300, 300)
org3.brain = CNNBrain(grid_size=20)

org4 = Organism(400, 400)
org4.brain = GeneticAlgorithmBrain()

org5 = Organism(150, 150)
org5.brain = NEATBrain()

# Simulation loop
for timestep in range(1000):
    for org in [org1, org2, org3, org4, org5]:
        # Get state
        state = org.get_state()

        # AI decides action
        action = org.brain.decide_action(state)

        # Move based on AI decision
        dx, dy = action['move_direction']
        org.x += dx * org.speed
        org.y += dy * org.speed

        # Calculate reward
        reward = org.brain.calculate_reward(old_state, state, action)

        # Learn
        org.brain.learn(old_state, action, reward, state, not org.alive)
```

---

## 🏆 AI Battle Arena - Who Wins?

### Test Scenarios:

1. **Survival Challenge**
   - 50 organisms, 5 of each brain type
   - Limited food
   - Temperature zones
   - Winner: Most survivors after 1000 timesteps

2. **Speed Test**
   - Which brain makes fastest decisions?
   - Winner: DQN (forward pass only)

3. **Learning Speed**
   - Which learns fastest?
   - Winner: Q-Learning (simple updates)

4. **Complex Environment**
   - Obstacles, temp zones, moving food
   - Winner: CNN or NEAT (spatial reasoning)

5. **Evolution Test**
   - 10 generations
   - Winner: CMA-ES or NEAT (best evolution)

---

## 🔬 Biological Inspiration

### Which Models Biologists Use:

1. **Genetic Algorithms** ✅
   - Evolution simulation
   - Population genetics
   - Natural selection studies

2. **NEAT** ✅
   - Brain evolution studies
   - Artificial life research
   - Behavioral ecology

3. **CMA-ES** ✅
   - Parameter estimation
   - Evolutionary dynamics
   - Optimization in biology

4. **CNN** ✅
   - Visual system modeling
   - Neural cortex simulation
   - Sensory processing

5. **Reinforcement Learning** ✅
   - Animal learning
   - Behavioral neuroscience
   - Dopamine reward systems

---

## 📁 File Structure

```
microlife/ml/
├── brain_base.py           → Base Brain interface
├── brain_rl.py             → Q-Learning, DQN, Double DQN
├── brain_cnn.py            → CNN, ResNet-CNN
└── brain_evolutionary.py   → GA, NEAT, CMA-ES
```

**Total:** 8 different AI models implemented!

---

## 🚀 Next: Create Battle Arena Demo

```bash
python demo_ai_battle.py
```

See all AI models compete in real-time!

---

## 🎓 Learning Paths

### Beginner:
1. Start with **Q-Learning** (simplest)
2. Try **Genetic Algorithm** (intuitive)

### Intermediate:
3. **DQN** (neural networks)
4. **CNN** (visual perception)

### Advanced:
5. **NEAT** (topology evolution)
6. **CMA-ES** (modern evolution)
7. **Double DQN** (state-of-the-art RL)

---

## 💡 Key Insights

### Learning-Based (RL):
- Learn during lifetime
- Adapt to environment changes
- Fast adaptation

### Evolution-Based (GA, NEAT, CMA-ES):
- Learn across generations
- Slower but more general
- Biological realism

### Hybrid (Best of both):
- Evolution for structure
- RL for weights
- Future work!

---

**Şimdi mikroorganizmalar gerçekten akıllı! 8 farklı AI modeli! 🧠🦠**

Hangisi en iyi? Test edin ve görün! 🏆
