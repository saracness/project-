# 🚀 QUICK START - One-Click Launch

## Run the Simulation in ONE CLICK!

### Windows Users 🪟
**Double-click:** `START_SIMULATION.bat`

### Linux/Mac Users 🐧🍎
**Double-click:** `START_SIMULATION.sh`

Or from terminal:
```bash
./START_SIMULATION.sh
```

### Any Platform (Python) 🐍
```bash
python START_SIMULATION.py
```

---

## What the Launcher Does Automatically

1. ✅ Checks Python version (needs 3.7+)
2. ✅ Installs missing dependencies (matplotlib, pandas)
3. ✅ Verifies all simulation files exist
4. ✅ Launches the Phase 2 demo with visualization
5. ✅ Saves data logs when you close the window

---

## What You'll See

```
🦠 Colored organisms (energy levels)
   - Red = Low energy (hungry!)
   - Yellow = High energy (well-fed)

🟢 Green dots = Food particles

🔴 Red circles = HOT temperature zones (drains energy)

🔵 Blue circles = COLD temperature zones (drains energy)

⬛ Gray rectangles = Obstacles (walls)

📊 Live statistics showing:
   - Population count
   - Seeking vs Wandering behaviors
   - Average energy levels
```

---

## Controls

- **Watch** the simulation run
- **Close window** to stop and save data
- Data saved to: `microlife/data/logs/`

---

## Troubleshooting

### "Python not found"
- Install Python 3.7+ from https://www.python.org/
- Make sure Python is in your PATH

### "Module not found"
- The launcher auto-installs dependencies
- Or manually run: `pip install matplotlib pandas`

### "Files not found"
- Make sure you're in the project directory
- Or clone and checkout the correct branch:
  ```bash
  git clone https://github.com/saracness/project-.git
  cd project-
  git checkout claude/microlife-ml-guide-011CUnQgJvemd2JyKLX8AkWK
  ```

---

## After Running

Your simulation data will be saved as CSV files in:
```
microlife/data/logs/
├── organism_logs_YYYYMMDD_HHMMSS.csv    ← Organism behaviors
├── timestep_logs_YYYYMMDD_HHMMSS.csv    ← Population stats
└── metadata_YYYYMMDD_HHMMSS.json        ← Simulation config
```

Ready for Phase 3 machine learning analysis! 📊

---

**That's it! Just one click to see intelligent organisms in action!** 🦠🧬
