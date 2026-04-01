# 🎨 Visual Guide to the Simulation

## What You'll See When You Run It

### Main Window
```
┌─────────────────────────────────────────────────────────┐
│  Micro-Life Simulation (Phase 2)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   🔴 [Hot Zone]        🟡🟡  ← Organisms               │
│                         🟢   ← Food                     │
│         🟡                                              │
│    🟢  🟡🟢                                            │
│              ⬛⬛⬛  ← Obstacle                         │
│    🟢           ⬛⬛⬛                                  │
│         🟡  🟢                                          │
│   🔵 [Cold Zone]    🟡                                 │
│              🟢  🟡                                     │
│                                                         │
│  ┌─────────────┐                                       │
│  │ Timestep: 245                                       │
│  │ Population: 15                                      │
│  │ Food: 28                                            │
│  │ Avg Energy: 112.3                                   │
│  │ Seeking: 6                                          │
│  │ Wandering: 9                                        │
│  └─────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 What Each Element Means

### Organisms (Colored Circles)
- **🟡 Yellow** = High energy (well-fed, healthy)
- **🟠 Orange** = Medium energy (doing okay)
- **🔴 Red** = Low energy (hungry, seeking food!)

Watch them change color as they eat and move!

### Food (Green Dots)
- **🟢 Green circles** = Food particles
- When an organism touches food, it disappears
- New food spawns automatically

### Temperature Zones
- **🔴 Red circle** = HOT zone (drains energy)
- **🔵 Blue circle** = COLD zone (drains energy)
- Semi-transparent circular areas
- Organisms avoid staying in them

### Obstacles
- **⬛ Gray rectangles** = Walls/blocks
- Organisms cannot pass through
- Creates maze-like challenges

### Organism Trails
- Faint colored lines behind organisms
- Shows recent movement path
- Helps visualize behavior patterns

---

## 📊 Statistics Panel (Top Left)

```
Timestep: 245        ← Current simulation step
Population: 15       ← Living organisms
Food: 28            ← Available food particles
Avg Energy: 112.3   ← Average energy level
Seeking: 6          ← Organisms actively hunting food
Wandering: 9        ← Organisms exploring randomly
```

---

## 🎭 Behaviors You'll Observe

### 1. **Food Seeking** (When Hungry)
```
   Organism (energy < 100)
        ↓
   [Scanning for food...]
        ↓
   [Food found within 100 units!]
        ↓
   [Moving directly toward it] →→→ 🟢
        ↓
   [Eaten! Energy +20]
```

**Visual:** Organism makes a **beeline** straight to the nearest food

### 2. **Wandering** (When Full)
```
   Organism (energy >= 100)
        ↓
   [Not hungry, exploring]
        ↓
   [Random movement] ⤴⤵↗↘
```

**Visual:** Organism moves in **random directions**

### 3. **Reproduction** (When Energy > 150)
```
   Organism (energy = 160)
        ↓
   [Enough energy to reproduce!]
        ↓
   [Creating offspring...]
        ↓
   🟡 → 🟡🟡  (parent + child)
```

**Visual:** A new organism appears **near the parent**

### 4. **Death** (When Energy = 0)
```
   Organism (energy = 5, 4, 3, 2, 1...)
        ↓
   [Energy depleted]
        ↓
   [Organism dies - disappears]
        ↓
   🟡 → ✖ (removed from screen)
```

**Visual:** Organism **fades away**

---

## 🔍 Watch For These Patterns

### Pattern 1: **The Hunt**
- Red organism (low energy) spots green food
- Tracks directly toward it in a **straight line**
- Statistics show "Seeking" count increases
- Organism turns yellow after eating

### Pattern 2: **Temperature Zone Avoidance**
- Organism enters red/blue zone
- Energy starts draining faster
- May seek food more urgently
- Sometimes dies if stuck too long

### Pattern 3: **Obstacle Navigation**
- Organism hits gray wall
- Bounces back slightly
- Changes direction
- Finds way around it

### Pattern 4: **Population Boom**
- Lots of food available
- Organisms gain energy
- Multiple reproductions happen
- Population spikes!

### Pattern 5: **Population Crash**
- Food runs out
- Energy depletes across population
- Many deaths in quick succession
- Only the luckiest survive

---

## ⏱️ Timeline of a Typical Run

```
0-100 steps:    Initial exploration
                Organisms scatter and find food
                Population stable at ~15-20

100-300 steps:  Growth phase
                Good food availability
                Some reproduction occurs
                Population: 20-30

300-500 steps:  Equilibrium
                Food consumption = food spawning
                Deaths = births
                Population stabilizes

500-800 steps:  Late game
                Possible food scarcity
                Strategic seeking becomes critical
                Survival of the fittest!
```

---

## 🎬 Real Example Scenarios

### Scenario A: "The Survivor"
```
Organism #7 (energy: 180, age: 450)

Timeline:
- Started in food-rich area
- Avoided temperature zones
- Reproduced 3 times
- Longest-living organism
- Final stats: 5 offspring, 450 timesteps survived
```

### Scenario B: "The Unlucky One"
```
Organism #3 (energy: 0, age: 45)

Timeline:
- Spawned in obstacle maze
- Couldn't find exit
- Got trapped in cold zone
- Energy drained rapidly
- Died at timestep 45
```

### Scenario C: "The Feast"
```
Timestep 200:
- 10 food particles spawn in cluster
- 5 organisms converge on the area
- All switch to "seeking" mode
- Mass feeding frenzy!
- 3 organisms reproduce afterward
```

---

## 🎓 Educational Observations

### For AI/ML Learning:
1. **Emergent Behavior** - Simple rules create complex patterns
2. **Decision Making** - Hunger drives intelligent choices
3. **Survival Strategies** - Different organisms take different paths
4. **Data Patterns** - Seeking vs wandering ratios change over time

### For Biology:
1. **Predator-Prey Dynamics** (organism-food relationship)
2. **Environmental Pressure** (temperature zones)
3. **Natural Selection** (only efficient seekers survive)
4. **Population Dynamics** (boom and bust cycles)

---

## 📸 What Gets Saved

When you close the window, the simulation saves:

```
microlife/data/logs/
├── organism_logs_20251104_125430.csv
│   → Every organism's position, energy, behavior
│   → Used for ML clustering analysis
│
├── timestep_logs_20251104_125430.csv
│   → Population statistics over time
│   → Used for trend analysis
│
└── metadata_20251104_125430.json
    → Simulation configuration
    → Temperature zones, obstacles, settings
```

---

## 🎮 Tips for Best Viewing

1. **Run for 500-800 timesteps** - See full life cycles
2. **Watch the statistics** - See behavior mode changes
3. **Look for trails** - Identify seeking vs wandering
4. **Count reproductions** - Watch population grow
5. **Observe deaths** - Learn what causes failure

---

## 🚀 Challenge Yourself

Try to answer these questions by watching:

1. What percentage of organisms are "seeking" on average?
2. How long does the average organism survive?
3. Which areas have the most deaths? (near obstacles? temp zones?)
4. Do organisms learn to avoid temperature zones? (No! They're not that smart yet - that's Phase 5!)
5. What's the maximum population you can sustain?

Answers are in the CSV data logs! 📊

---

**Ready to watch artificial life unfold? Just double-click the launcher!** 🦠✨
