# 🎛️ Hyperparameter Tuning Guide
## Yapay Zeka Ayarlarını Değiştirme Rehberi

---

## 🎯 Hyperparameter Nedir?

**Hyperparameter:** Yapay zeka modelinin davranışını kontrol eden ayarlardır.

Örnek:
- Öğrenme hızı (learning rate)
- Exploration oranı (epsilon)
- Network boyutu (hidden_size)
- Mutation oranı (mutation_rate)

**Bunları değiştirerek AI'nın performansını optimize edebilirsiniz!**

---

## 📊 TÜM AI MODELLERİNİN HYPERPARAMETRELERİ

### 1. Q-Learning Brain

**Dosya:** `microlife/ml/brain_rl.py` → `QLearningBrain`

```python
from microlife.ml.brain_rl import QLearningBrain

brain = QLearningBrain(
    learning_rate=0.1,      # 🎛️ Öğrenme hızı
    discount_factor=0.95,   # 🎛️ Gelecek ödül değeri
    epsilon=0.3             # 🎛️ Exploration oranı
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `learning_rate` | 0.1 | 0.01-0.5 | Q-değer güncelleme hızı. **Yüksek** = hızlı ama kararsız, **Düşük** = yavaş ama stabil |
| `discount_factor` | 0.95 | 0.8-0.99 | Gelecekteki ödüllerin değeri. **Yüksek** = uzun vadeli düşünür, **Düşük** = kısa vadeli |
| `epsilon` | 0.3 | 0.1-0.9 | Keşif (exploration) oranı. **Yüksek** = daha çok rastgele, **Düşük** = daha çok öğrenileni kullan |

**Nasıl Değiştirirsiniz:**
```python
# Hızlı öğrenme için
brain = QLearningBrain(learning_rate=0.3, epsilon=0.5)

# Dikkatli, uzun vadeli düşünen için
brain = QLearningBrain(learning_rate=0.05, discount_factor=0.99, epsilon=0.1)
```

---

### 2. DQN Brain (Deep Q-Network)

**Dosya:** `microlife/ml/brain_rl.py` → `DQNBrain`

```python
from microlife.ml.brain_rl import DQNBrain

brain = DQNBrain(
    state_size=7,           # 🎛️ State boyutu (değiştirme!)
    hidden_size=32,         # 🎛️ Hidden layer nöron sayısı
    learning_rate=0.001     # 🎛️ Öğrenme hızı
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `state_size` | 7 | Sabit | State vektör boyutu (değiştirmeyin) |
| `hidden_size` | 32 | 16-128 | Hidden layer nöron sayısı. **Büyük** = daha karmaşık, **Küçük** = daha hızlı |
| `learning_rate` | 0.001 | 0.0001-0.01 | Neural network öğrenme hızı. **Yüksek** = hızlı ama kararsız |
| `batch_size` | 32 | 16-128 | Experience replay batch boyutu |
| `epsilon` | 0.5 | 0.1-0.9 | Exploration oranı (başlangıç) |
| `gamma` | 0.95 | 0.8-0.99 | Discount factor |

**Nasıl Değiştirirsiniz:**
```python
# Büyük, güçlü network
brain = DQNBrain(state_size=7, hidden_size=64, learning_rate=0.001)

# Küçük, hızlı network
brain = DQNBrain(state_size=7, hidden_size=16, learning_rate=0.005)

# İçerideki parametreler
brain.batch_size = 64  # Daha büyük batch
brain.epsilon = 0.7    # Daha çok exploration
```

---

### 3. Double DQN Brain

**Dosya:** `microlife/ml/brain_rl.py` → `DoubleDQNBrain`

```python
from microlife.ml.brain_rl import DoubleDQNBrain

brain = DoubleDQNBrain(
    state_size=7,
    hidden_size=32,
    learning_rate=0.001
)

# Ek parametreler (içeride)
brain.update_target_every = 100  # 🎛️ Target network güncelleme sıklığı
```

#### Hyperparameters:

DQN ile aynı + ek:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `update_target_every` | 100 | 50-500 | Target network kaç adımda bir güncellenir. **Yüksek** = daha stabil |

**Nasıl Değiştirirsiniz:**
```python
brain = DoubleDQNBrain(hidden_size=48)
brain.update_target_every = 200  # Daha stabil
brain.learning_rate = 0.0005     # Daha yavaş öğrenme
```

---

### 4. CNN Brain (Convolutional Neural Network)

**Dosya:** `microlife/ml/brain_cnn.py` → `CNNBrain`

```python
from microlife.ml.brain_cnn import CNNBrain

brain = CNNBrain(
    grid_size=20,           # 🎛️ Visual grid boyutu
    hidden_size=64          # 🎛️ FC layer boyutu
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `grid_size` | 20 | 10-50 | Visual perception grid boyutu. **Büyük** = daha detaylı görüş ama yavaş |
| `hidden_size` | 64 | 32-128 | Fully connected layer boyutu |
| `epsilon` | 0.3 | 0.1-0.7 | Exploration oranı |
| `lr` | 0.001 | 0.0001-0.01 | Learning rate |

**Nasıl Değiştirirsiniz:**
```python
# Yüksek çözünürlük görüş
brain = CNNBrain(grid_size=30, hidden_size=96)

# Hızlı ama düşük çözünürlük
brain = CNNBrain(grid_size=15, hidden_size=48)

# Parametreler
brain.epsilon = 0.2  # Az exploration
brain.lr = 0.002     # Hızlı öğrenme
```

---

### 5. ResNet-CNN Brain

**Dosya:** `microlife/ml/brain_cnn.py` → `ResidualCNNBrain`

CNN ile aynı parametreler + residual connections.

```python
from microlife.ml.brain_cnn import ResidualCNNBrain

brain = ResidualCNNBrain(
    grid_size=20,
    hidden_size=64
)
```

---

### 6. Genetic Algorithm Brain

**Dosya:** `microlife/ml/brain_evolutionary.py` → `GeneticAlgorithmBrain`

```python
from microlife.ml.brain_evolutionary import GeneticAlgorithmBrain

brain = GeneticAlgorithmBrain(
    genome_size=20,         # 🎛️ Gen sayısı
    mutation_rate=0.1       # 🎛️ Mutasyon oranı
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `genome_size` | 20 | 10-50 | Genom boyutu (gen sayısı). **Büyük** = daha karmaşık davranış |
| `mutation_rate` | 0.1 | 0.01-0.5 | Mutasyon olasılığı. **Yüksek** = daha çok değişim |

**Nasıl Değiştirirsiniz:**
```python
# Basit, stabil genom
brain = GeneticAlgorithmBrain(genome_size=15, mutation_rate=0.05)

# Karmaşık, hızlı evrim
brain = GeneticAlgorithmBrain(genome_size=30, mutation_rate=0.2)
```

**Evolution için:**
```python
# Mutasyon
brain.mutation_rate = 0.15  # Değiştir
brain.mutate()              # Uygula

# Crossover
child = brain1.crossover(brain2)
```

---

### 7. NEAT Brain (NeuroEvolution)

**Dosya:** `microlife/ml/brain_evolutionary.py` → `NEATBrain`

```python
from microlife.ml.brain_evolutionary import NEATBrain

brain = NEATBrain(
    input_size=7,           # 🎛️ Input nöron sayısı
    output_size=9           # 🎛️ Output nöron sayısı
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `input_size` | 7 | Sabit | Input boyutu |
| `output_size` | 9 | Sabit | Output boyutu |

**Mutation parametreleri:**
```python
brain.mutate(
    add_node_prob=0.03,     # 🎛️ Yeni nöron ekleme olasılığı
    add_conn_prob=0.05,     # 🎛️ Yeni bağlantı ekleme olasılığı
    weight_mut_prob=0.8     # 🎛️ Weight mutasyon olasılığı
)
```

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `add_node_prob` | 0.03 | 0.01-0.1 | Yeni hidden nöron ekleme olasılığı. **Yüksek** = hızlı karmaşıklaşma |
| `add_conn_prob` | 0.05 | 0.01-0.2 | Yeni bağlantı ekleme olasılığı |
| `weight_mut_prob` | 0.8 | 0.5-0.95 | Weight değişikliği olasılığı |

**Nasıl Değiştirirsiniz:**
```python
# Yavaş, dikkatli evrim
brain.mutate(add_node_prob=0.01, add_conn_prob=0.02, weight_mut_prob=0.9)

# Hızlı, agresif evrim
brain.mutate(add_node_prob=0.08, add_conn_prob=0.15, weight_mut_prob=0.7)
```

---

### 8. CMA-ES Brain (Evolution Strategy)

**Dosya:** `microlife/ml/brain_evolutionary.py` → `CMAESBrain`

```python
from microlife.ml.brain_evolutionary import CMAESBrain

brain = CMAESBrain(
    param_size=20           # 🎛️ Parametre sayısı
)
```

#### Hyperparameters:

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `param_size` | 20 | 10-50 | Parametre vektör boyutu |
| `sigma` | 1.0 | 0.1-5.0 | Step size (otomatik adapte olur) |

**Nasıl Değiştirirsiniz:**
```python
# Büyük parametre space
brain = CMAESBrain(param_size=30)
brain.sigma = 1.5  # Büyük adımlar

# Küçük, hassas arama
brain = CMAESBrain(param_size=15)
brain.sigma = 0.5  # Küçük adımlar
```

---

## 🎮 DEMO'DA NASIL DEĞİŞTİRİLİR?

### demo_ai_battle.py İçinde:

```python
# Dosyayı aç: demo_ai_battle.py
# create_ai_organisms() fonksiyonunu bul

def create_ai_organisms(environment):
    organisms = []

    # Q-Learning parametrelerini değiştir
    brain = QLearningBrain(
        learning_rate=0.2,      # ← Buradan değiştir!
        epsilon=0.5             # ← Buradan değiştir!
    )

    # DQN parametrelerini değiştir
    brain = DQNBrain(
        hidden_size=48,         # ← Buradan değiştir!
        learning_rate=0.002     # ← Buradan değiştir!
    )

    # CNN parametrelerini değiştir
    brain = CNNBrain(
        grid_size=25,           # ← Buradan değiştir!
        hidden_size=96          # ← Buradan değiştir!
    )

    # Genetic Algorithm parametrelerini değiştir
    brain = GeneticAlgorithmBrain(
        genome_size=25,         # ← Buradan değiştir!
        mutation_rate=0.15      # ← Buradan değiştir!
    )

    # NEAT mutasyon parametreleri
    # İçeride brain.mutate() çağrısına parametre ekle

    return organisms
```

---

## 📈 PARAMETRE ETKİLERİ

### Learning Rate (Öğrenme Hızı)

```
Çok Düşük (0.01):
├─ Avantaj: Çok stabil
└─ Dezavantaj: ÇOK yavaş öğrenir

İyi (0.1):
├─ Avantaj: Dengeli
└─ Dezavantaj: Bazen yavaş

Yüksek (0.5):
├─ Avantaj: Hızlı öğrenir
└─ Dezavantaj: Kararsız, overfitting

Çok Yüksek (1.0):
├─ Avantaj: -
└─ Dezavantaj: Hiç öğrenemez, kaotik
```

### Epsilon (Exploration)

```
Düşük (0.1):
├─ Davranış: Hep aynı stratejileri kullanır
├─ Avantaj: Öğrendiğini exploit eder
└─ Dezavantaj: Yeni şeyler keşfedemez

Orta (0.3):
├─ Davranış: Dengeli keşif
├─ Avantaj: Hem öğrenir hem keşfeder
└─ Dezavantaj: -

Yüksek (0.7):
├─ Davranış: Sürekli deneme yanılma
├─ Avantaj: Çok keşif yapar
└─ Dezavantaj: Öğrendiğini kullanamaz
```

### Hidden Size (Network Boyutu)

```
Küçük (16):
├─ Avantaj: Hızlı, az bellek
└─ Dezavantaj: Karmaşık patterns öğrenemez

Orta (32-48):
├─ Avantaj: Dengeli
└─ Dezavantaj: -

Büyük (64-128):
├─ Avantaj: Karmaşık patterns
└─ Dezavantaj: Yavaş, overfitting riski
```

### Mutation Rate (Mutasyon Oranı)

```
Düşük (0.05):
├─ Evrim: Yavaş, dikkatli
├─ Avantaj: Stabil
└─ Dezavantaj: Yavaş adapte olur

Orta (0.1):
├─ Evrim: Dengeli
└─ Avantaj: İyi evrim hızı

Yüksek (0.3):
├─ Evrim: Hızlı, radikal
├─ Avantaj: Hızlı değişim
└─ Dezavantaj: İyi genler kaybolabilir
```

---

## 🧪 DENEY ÖNERİLERİ

### Deney 1: Hızlı Öğrenen vs Dikkatli

```python
# Hızlı öğrenen (agresif)
brain1 = DQNBrain(learning_rate=0.01, hidden_size=64)
brain1.epsilon = 0.7

# Dikkatli öğrenen (muhafazakar)
brain2 = DQNBrain(learning_rate=0.0005, hidden_size=32)
brain2.epsilon = 0.2

# Hangisi daha iyi?
```

### Deney 2: Keşifçi vs Sömürücü

```python
# Explorer (keşfeder)
brain1 = QLearningBrain(epsilon=0.8)

# Exploiter (sömürür)
brain2 = QLearningBrain(epsilon=0.1)

# Hangisi daha uzun yaşar?
```

### Deney 3: Basit vs Karmaşık Network

```python
# Basit
brain1 = DQNBrain(hidden_size=16)

# Karmaşık
brain2 = DQNBrain(hidden_size=128)

# Hangisi daha iyi karar verir?
```

### Deney 4: Yavaş vs Hızlı Evrim

```python
# Yavaş evrim
brain1 = GeneticAlgorithmBrain(mutation_rate=0.03)

# Hızlı evrim
brain2 = GeneticAlgorithmBrain(mutation_rate=0.25)

# Hangisi daha iyi adapte olur?
```

---

## 💡 GENEL TAVSİYELER

### 1. Başlangıç İçin:
```python
# En iyi başlangıç değerleri
QLearningBrain(learning_rate=0.1, epsilon=0.3)
DQNBrain(hidden_size=32, learning_rate=0.001)
GeneticAlgorithmBrain(genome_size=20, mutation_rate=0.1)
```

### 2. Hızlı Sonuç İstiyorsanız:
```python
# Hızlı öğrenme
learning_rate = 0.3
epsilon = 0.5
mutation_rate = 0.2
```

### 3. Stabil Öğrenme İstiyorsanız:
```python
# Yavaş ama güvenli
learning_rate = 0.05
epsilon = 0.2
mutation_rate = 0.05
```

### 4. Karmaşık Ortam İçin:
```python
# Büyük network
hidden_size = 64
grid_size = 25
genome_size = 30
```

---

## 📋 PARAMETER CHEAT SHEET

| Parametre | Küçük Değer → Büyük Değer |
|-----------|---------------------------|
| `learning_rate` | Yavaş öğrenme ↔️ Hızlı öğrenme |
| `epsilon` | Öğrenileni kullan ↔️ Keşfet |
| `discount_factor` | Kısa vade ↔️ Uzun vade |
| `hidden_size` | Basit ↔️ Karmaşık |
| `mutation_rate` | Stabil ↔️ Değişken |
| `grid_size` | Hızlı ↔️ Detaylı |

---

## 🚀 HIZLI BAŞLANGIÇ

### Adım 1: Dosyayı Aç
```bash
# Editör ile aç
nano demo_ai_battle.py
# veya
code demo_ai_battle.py
```

### Adım 2: create_ai_organisms() Fonksiyonunu Bul
```python
def create_ai_organisms(environment):
    # Burası!
```

### Adım 3: Parametreleri Değiştir
```python
# Örnek: Q-Learning'i hızlandır
brain = QLearningBrain(
    learning_rate=0.3,  # 0.1 yerine 0.3
    epsilon=0.5         # 0.3 yerine 0.5
)
```

### Adım 4: Kaydet ve Çalıştır
```bash
python demo_ai_battle.py
```

### Adım 5: Gözlemle ve Karşılaştır!

---

## 🎯 SONUÇ

**Hyperparameter tuning = AI'nın kişiliğini değiştirmek!**

- **learning_rate** → Ne kadar hızlı öğrensin?
- **epsilon** → Ne kadar keşfetsin?
- **hidden_size** → Ne kadar karmaşık düşünsün?
- **mutation_rate** → Ne kadar değişsin?

**Kendiniz deneyin ve en iyi kombinasyonu bulun!** 🎛️🧠🚀

---

## 📖 Daha Fazla Bilgi:

- `AI_BRAINS_GUIDE.md` → Her AI'ın detaylı açıklaması
- `demo_ai_battle.py` → Parametreleri değiştirme yeri
- `microlife/ml/brain_*.py` → Kaynak kodlar

**İyi deneyler! 🧪**
