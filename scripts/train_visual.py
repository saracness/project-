"""
Live visualizer — every pixel is driven by the real RL brain.

Each frame:
  1. brain.decide_action(state)   → picks action (Q-values computed)
  2. env.step(action_idx)         → organism moves, eats, survives
  3. pygame renders the result    → you see what the brain decided

Live parameter editor lets you change epsilon, lr, gamma, personality,
and speed IN REAL TIME while the simulation is running.

Controls
--------
  SPACE       pause / resume
  TAB         cycle selected parameter in editor
  LEFT/RIGHT  decrease / increase selected parameter
  +/-         speed up / slow down (shortcut)
  R           restart episode
  N           cycle NeuroBrain personality
  Q           quit and save summary

Usage
-----
  python scripts/train_visual.py --brain dqn --fps 30 --episodes 100
  python scripts/train_visual.py --brain neuro --personality dopaminergic --episodes 100
  python scripts/train_visual.py --brain neuro --headless --episodes 200 --out outputs/viz
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.gym_env import MicrolifeEnv, ACTIONS, N_ACTIONS
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain
from microlife.ml.brain_neuro import NeuroBrain, SUPPORTED_PERSONALITIES

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
SIM_W, SIM_H = 600, 600
PANEL_W      = 380          # wider panel for live editor
WIN_W        = SIM_W + PANEL_W
WIN_H        = SIM_H
SCALE        = SIM_W / 300  # world 300×300 → screen pixels

# Colours
BG        = (15,  20,  30)
GRID_C    = (25,  35,  50)
FOOD_C    = (80, 220,  80)
ORG_C     = (60, 180, 255)
TRAIL_C   = (40, 100, 160)
PANEL_BG  = (20,  25,  38)
TEXT_C    = (200, 210, 230)
DIM_C     = (90, 100, 120)
BAR_POS   = (80, 180,  80)
BAR_NEG   = (220, 80,  60)
ARROW_C   = (255, 230,  50)
BEST_C    = (255, 100, 255)
SEL_C     = (60, 160, 255)    # selected param highlight
NEURO_C   = (180, 120, 255)   # NeuroBrain accent colour

PERSONALITIES = sorted(SUPPORTED_PERSONALITIES)   # alphabetical list


# ---------------------------------------------------------------------------
# Brain factory
# ---------------------------------------------------------------------------

def make_brain(name: str, personality: str = "dopaminergic"):
    name = name.lower()
    if name == "qlearning":
        return QLearningBrain(learning_rate=0.1, discount_factor=0.95, epsilon=0.5)
    if name == "dqn":
        return DQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    if name == "ddqn":
        return DoubleDQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    if name == "neuro":
        return NeuroBrain(personality_type=personality,
                          state_size=8, hidden_size=64, learning_rate=0.001)
    raise ValueError(f"Unknown brain '{name}'. Choose: qlearning dqn ddqn neuro")


# ---------------------------------------------------------------------------
# Q-value extraction
# ---------------------------------------------------------------------------

def get_q_values(brain, state: dict) -> np.ndarray:
    if isinstance(brain, DQNBrain):        # NeuroBrain inherits DQNBrain
        return brain._forward(brain.get_state_vector(state))
    if isinstance(brain, QLearningBrain):
        disc = brain._discretize_state(state)
        return brain.q_table.get(disc, np.zeros(N_ACTIONS)).copy()
    return np.zeros(N_ACTIONS)


# ---------------------------------------------------------------------------
# Live parameter editor
# ---------------------------------------------------------------------------

class ParamEditor:
    """
    Manages a list of editable hyperparameters and applies changes to the brain.

    Parameters are rendered as rows in the panel. TAB cycles the selected row;
    LEFT / RIGHT change the value. Changes take effect immediately.
    """

    # (label, attr on brain, min, max, step, type)
    # type: "float" | "int" | "choice"
    PARAMS = [
        ("ε Epsilon",   "epsilon",  0.0,    1.0,   0.01,   "float"),
        ("α LR",        "lr",       1e-4,   0.05,  5e-4,   "float"),
        ("γ Gamma",     "gamma",    0.50,   0.999, 0.01,   "float"),
        ("Personality", None,       None,   None,  None,   "choice"),
        ("Speed",       None,       1,      30,    1,      "int"),
    ]

    def __init__(self, brain, speed_ref: list[int]):
        self.brain      = brain
        self.speed_ref  = speed_ref   # mutable list[int] shared with main loop
        self._sel       = 0           # selected row index

    def handle_key(self, key) -> bool:
        """Return True if event was consumed."""
        if key == pygame.K_TAB:
            self._sel = (self._sel + 1) % len(self.PARAMS)
            return True
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            self._adjust(key == pygame.K_RIGHT)
            return True
        if key == pygame.K_n:
            self._cycle_personality(+1)
            return True
        return False

    def _adjust(self, increase: bool) -> None:
        label, attr, lo, hi, step, kind = self.PARAMS[self._sel]
        sign = 1 if increase else -1

        if kind == "choice":
            self._cycle_personality(sign)
            return

        if kind == "int" and attr is None:
            # Speed
            self.speed_ref[0] = int(
                max(lo, min(hi, self.speed_ref[0] + sign * step))
            )
            return

        if attr and hasattr(self.brain, attr):
            cur = getattr(self.brain, attr)
            new = float(np.clip(cur + sign * step, lo, hi))
            setattr(self.brain, attr, round(new, 6))

    def _cycle_personality(self, direction: int) -> None:
        if not isinstance(self.brain, NeuroBrain):
            return
        idx = PERSONALITIES.index(self.brain.personality_type)
        new_idx = (idx + direction) % len(PERSONALITIES)
        self.brain.set_personality(PERSONALITIES[new_idx])

    def draw(self, surf: pygame.Surface, font_sm, x0: int, y0: int,
             panel_w: int) -> int:
        """Draw editor rows. Returns y position after last row."""
        bar_max = panel_w - 110

        for i, (label, attr, lo, hi, step, kind) in enumerate(self.PARAMS):
            selected = (i == self._sel)
            row_col = SEL_C if selected else DIM_C

            # Bullet
            bullet = "●" if selected else "○"
            lbl = font_sm.render(f"{bullet} {label:<12}", True, row_col)
            surf.blit(lbl, (x0, y0))

            # Value / slider
            if kind == "choice":
                p_name = (self.brain.personality_type
                          if isinstance(self.brain, NeuroBrain) else "N/A")
                val_s = f"{p_name:<16} ◄►"
                val_lbl = font_sm.render(val_s, True, NEURO_C if selected else row_col)
                surf.blit(val_lbl, (x0 + 115, y0))

            elif kind == "int" and attr is None:
                # Speed
                val = self.speed_ref[0]
                frac = (val - lo) / (hi - lo)
                pygame.draw.rect(surf, (40, 50, 70),
                                 (x0 + 115, y0 + 2, bar_max, 10))
                pygame.draw.rect(surf, row_col,
                                 (x0 + 115, y0 + 2, int(frac * bar_max), 10))
                val_lbl = font_sm.render(f" {val}  ◄►", True, row_col)
                surf.blit(val_lbl, (x0 + 115 + bar_max + 2, y0))

            else:
                cur = getattr(self.brain, attr, 0.0) if attr else 0.0
                frac = (cur - lo) / (hi - lo) if hi != lo else 0.0
                frac = max(0.0, min(1.0, frac))
                pygame.draw.rect(surf, (40, 50, 70),
                                 (x0 + 115, y0 + 2, bar_max, 10))
                pygame.draw.rect(surf, row_col,
                                 (x0 + 115, y0 + 2, int(frac * bar_max), 10))
                fmt = f"{cur:.4f}" if kind == "float" else f"{int(cur)}"
                val_lbl = font_sm.render(f" {fmt}  ◄►", True, row_col)
                surf.blit(val_lbl, (x0 + 115 + bar_max + 2, y0))

            y0 += 18

        return y0


# ---------------------------------------------------------------------------
# Coordinate + drawing helpers
# ---------------------------------------------------------------------------

def _ws(wx, wy):   # world → screen
    return int(wx * SCALE), int(wy * SCALE)


def draw_grid(surf):
    for x in range(0, SIM_W, 50):
        pygame.draw.line(surf, GRID_C, (x, 0), (x, SIM_H))
    for y in range(0, SIM_H, 50):
        pygame.draw.line(surf, GRID_C, (0, y), (SIM_W, y))


def draw_temp_zones(surf, env):
    overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    for z in env._env.temperature_zones:
        col = (200, 60, 40, 55) if z.temperature > 0 else (60, 120, 220, 55)
        pygame.draw.circle(overlay, col, _ws(z.x, z.y), int(z.radius * SCALE))
    surf.blit(overlay, (0, 0))


def draw_food(surf, env):
    r = max(3, int(2 * SCALE))
    for f in env._env.food_particles:
        if not f.consumed:
            pygame.draw.circle(surf, FOOD_C, _ws(f.x, f.y), r)


def draw_organism(surf, env, q_values: np.ndarray):
    o = env._organism
    if not o.alive:
        return
    # Trail
    for i in range(1, len(o.trail[-30:])):
        t = o.trail[-30:]
        pygame.draw.line(surf, TRAIL_C, _ws(*t[i-1]), _ws(*t[i]), 1)
    ox, oy = _ws(o.x, o.y)
    radius = max(6, int(o.size * SCALE))
    er = max(0.0, min(1.0, o.energy / 200.0))
    pygame.draw.circle(surf, (int(255*(1-er)), int(255*er), 80),
                       (ox, oy), radius + 3, 2)
    pygame.draw.circle(surf, ORG_C, (ox, oy), radius)
    best = int(np.argmax(q_values))
    q_min, q_max = q_values.min(), q_values.max()
    q_range = q_max - q_min if q_max != q_min else 1.0
    for i, (dx, dy) in enumerate(ACTIONS[:-1]):
        norm = (q_values[i] - q_min) / q_range
        length = int(norm * 28 + 4)
        ang = math.atan2(-dy, dx)
        ex = ox + int(math.cos(ang) * length)
        ey = oy + int(math.sin(ang) * length)
        pygame.draw.line(surf, BEST_C if i == best else DIM_C,
                         (ox, oy), (ex, ey), 2 if i == best else 1)
    if best < 8:
        dx, dy = ACTIONS[best]
        ang = math.atan2(-dy, dx)
        pygame.draw.circle(surf, BEST_C,
                           (ox + int(math.cos(ang)*24),
                            oy + int(math.sin(ang)*24)), 3)


def draw_panel(surf, font_lg, font_sm, brain, env, ep, total_ep,
               ep_reward, step, q_values, action_idx,
               fps, paused, editor: ParamEditor):
    surf.fill(PANEL_BG)
    y = 10
    pw = PANEL_W

    def txt(text, f=None, col=TEXT_C):
        nonlocal y
        lbl = (f or font_sm).render(text, True, col)
        surf.blit(lbl, (10, y))
        y += lbl.get_height() + 3

    def sep(label=""):
        nonlocal y
        if label:
            txt(f"─── {label}", col=DIM_C)
        else:
            y += 4

    # Header
    brain_label = (brain.brain_type if hasattr(brain, "brain_type")
                   else type(brain).__name__)
    txt(f"MicrolifeEnv  —  {brain_label}", font_lg)
    txt(f"Episode {ep}/{total_ep}    FPS {fps:.0f}    step {step}", col=DIM_C)
    if paused:
        txt("⏸  PAUSED  (SPACE to resume)", col=(255, 200, 50))
    sep()

    # NeuroBrain personality banner
    if isinstance(brain, NeuroBrain):
        info = brain.get_neuro_info()
        txt(f"Neuron: {info['personality'].upper()}", col=NEURO_C)
        txt(f" DA={info['da_sensitivity']:.2f}  "
            f"5HT={info['5ht_sensitivity']:.2f}  "
            f"ACh={info['ach_sensitivity']:.2f}",
            col=NEURO_C)
        if "novel_cells" in info:
            txt(f" Novel cells explored: {info['novel_cells']}", col=NEURO_C)
        sep()

    # Organism
    sep("Organism")
    o = env._organism
    ep_pct = int(o.energy / 200 * 100) if o.alive else 0
    txt(f"Energy : {o.energy:6.1f}  ({ep_pct}%)")
    txt(f"Age    : {o.age:6d}    Alive: {'YES' if o.alive else 'NO'}")
    txt(f"Reward : {ep_reward:+.2f}")
    sep()

    # Brain
    sep("Brain")
    txt(f"ε={getattr(brain,'epsilon',0):.4f}  "
        f"lr={getattr(brain,'lr',0):.5f}  "
        f"γ={getattr(brain,'gamma',0):.3f}")
    txt(f"Decisions : {brain.decision_count}   "
        f"Total reward : {brain.total_reward:.1f}")
    if hasattr(brain, "memory"):
        txt(f"Replay buffer: {len(brain.memory)}")
    sep()

    # Q-values
    sep("Q-values (this state)")
    ANAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Stay"]
    bar_w = pw - 82
    q_min, q_max = q_values.min(), q_values.max()
    qr = q_max - q_min if q_max != q_min else 1.0
    best = int(np.argmax(q_values))
    for i, (name, qv) in enumerate(zip(ANAMES, q_values)):
        bw = int((qv - q_min) / qr * bar_w)
        col = BEST_C if i == best else (BAR_POS if qv >= 0 else BAR_NEG)
        surf.blit(font_sm.render(f"{name:>4}", True,
                                 TEXT_C if i == best else DIM_C), (10, y))
        pygame.draw.rect(surf, col, (50, y+2, max(1, bw), 10))
        surf.blit(font_sm.render(f"{qv:+.3f}", True,
                                 TEXT_C if i == best else DIM_C),
                  (50 + bar_w + 4, y))
        y += 16
    txt(f"→ Chose: {ANAMES[action_idx]}", col=ARROW_C)
    sep()

    # Live editor
    sep("Live Edit  (TAB=select  ◄► =change)")
    y = editor.draw(surf, font_sm, 10, y, pw - 20)
    sep()

    # Controls
    sep("Controls")
    for line in ["SPACE pause  R restart  Q quit",
                 "N cycle personality"]:
        txt(line, col=DIM_C)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(
    brain_name: str = "dqn",
    personality: str = "dopaminergic",
    total_episodes: int = 50,
    max_steps: int = 500,
    target_fps: int = 30,
    seed: int | None = 42,
    headless: bool = False,
    out_dir: str = "outputs/viz",
):
    os.makedirs(out_dir, exist_ok=True)

    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()
    win   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("MicrolifeEnv — Live RL Visualizer")
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm = pygame.font.SysFont("monospace", 12)
    sim_surf   = pygame.Surface((SIM_W, SIM_H))
    panel_surf = pygame.Surface((PANEL_W, WIN_H))

    env   = MicrolifeEnv(width=300, height=300, n_food=20,
                         n_temp_zones=3, max_steps=max_steps, seed=seed)
    brain = make_brain(brain_name, personality)

    speed_ref = [1]   # mutable, shared with ParamEditor
    editor    = ParamEditor(brain, speed_ref)
    paused    = False
    ep_rewards: list[float] = []

    for ep in range(1, total_episodes + 1):
        obs        = env.reset()
        state      = env.obs_to_brain_state(obs)
        ep_rew     = 0.0
        step       = 0
        q_values   = np.zeros(N_ACTIONS)
        action_idx = 8
        info: dict = {}

        if headless:
            # Full-speed, no event loop
            while True:
                if env._step_count >= max_steps or not env._organism.alive:
                    break
                action_dict = brain.decide_action(state)
                action_idx  = action_dict["_action_idx"]
                q_values    = get_q_values(brain, state)
                obs_next, reward, done, info = env.step(action_idx)
                next_state  = env.obs_to_brain_state(obs_next)
                brain.learn(state, action_dict, reward, next_state, done)
                ep_rew += reward
                step   += 1
                state   = next_state
                if done:
                    break
            _render(win, sim_surf, panel_surf, font_lg, font_sm,
                    env, brain, ep, total_episodes, ep_rew,
                    step, q_values, action_idx, 0.0, paused, editor)
            if ep % 10 == 0 or ep == 1:
                path = os.path.join(out_dir, f"ep_{ep:04d}.png")
                pygame.image.save(win, path)
                print(f"Ep {ep:4d}/{total_episodes} | "
                      f"reward={ep_rew:7.2f} | "
                      f"energy={info.get('energy',0):6.1f} | "
                      f"ε={getattr(brain,'epsilon',0):.3f} | "
                      f"saved {path}")

        else:
            done_ep = False
            while not done_ep:
                # Events
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        _save_summary(ep_rewards, out_dir)
                        pygame.quit()
                        return ep_rewards
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_SPACE:
                            paused = not paused
                        elif ev.key == pygame.K_r:
                            done_ep = True
                        elif ev.key == pygame.K_q:
                            _save_summary(ep_rewards, out_dir)
                            pygame.quit()
                            return ep_rewards
                        elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                        pygame.K_KP_PLUS):
                            speed_ref[0] = min(speed_ref[0] + 1, 30)
                        elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            speed_ref[0] = max(speed_ref[0] - 1, 1)
                        else:
                            editor.handle_key(ev.key)

                if paused:
                    _render(win, sim_surf, panel_surf, font_lg, font_sm,
                            env, brain, ep, total_episodes, ep_rew,
                            step, q_values, action_idx,
                            clock.get_fps(), True, editor)
                    clock.tick(30)
                    continue

                for _ in range(speed_ref[0]):
                    if env._step_count >= max_steps or not env._organism.alive:
                        done_ep = True
                        break
                    action_dict = brain.decide_action(state)
                    action_idx  = action_dict["_action_idx"]
                    q_values    = get_q_values(brain, state)
                    obs_next, reward, done, info = env.step(action_idx)
                    next_state  = env.obs_to_brain_state(obs_next)
                    brain.learn(state, action_dict, reward, next_state, done)
                    ep_rew += reward
                    step   += 1
                    state   = next_state
                    if done:
                        done_ep = True
                        break

                _render(win, sim_surf, panel_surf, font_lg, font_sm,
                        env, brain, ep, total_episodes, ep_rew,
                        step, q_values, action_idx,
                        clock.get_fps(), False, editor)
                clock.tick(target_fps)

        ep_rewards.append(ep_rew)

    _save_summary(ep_rewards, out_dir)
    pygame.quit()
    return ep_rewards


def _render(win, sim_surf, panel_surf, font_lg, font_sm,
            env, brain, ep, total_ep, ep_rew, step,
            q_values, action_idx, fps, paused, editor):
    sim_surf.fill(BG)
    draw_grid(sim_surf)
    draw_temp_zones(sim_surf, env)
    draw_food(sim_surf, env)
    draw_organism(sim_surf, env, q_values)
    draw_panel(panel_surf, font_lg, font_sm, brain, env,
               ep, total_ep, ep_rew, step,
               q_values, action_idx, fps, paused, editor)
    win.blit(sim_surf, (0, 0))
    win.blit(panel_surf, (SIM_W, 0))
    pygame.display.flip()


def _save_summary(ep_rewards: list, out_dir: str):
    if not ep_rewards:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep_rewards, alpha=0.4, color="steelblue")
        window = min(20, len(ep_rewards))
        if len(ep_rewards) >= window:
            roll = np.convolve(ep_rewards, np.ones(window)/window, "valid")
            ax.plot(range(window-1, len(ep_rewards)), roll,
                    color="orange", lw=2, label=f"{window}-ep mean")
        ax.set(xlabel="Episode", ylabel="Total reward",
               title="MicrolifeEnv — RL learning curve")
        ax.legend(); ax.grid(alpha=0.3)
        path = os.path.join(out_dir, "learning_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Learning curve saved → {path}")
    except Exception as e:
        print(f"Could not save summary: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Live RL visualizer for MicrolifeEnv")
    p.add_argument("--brain",       default="dqn",
                   choices=["qlearning", "dqn", "ddqn", "neuro"])
    p.add_argument("--personality", default="dopaminergic",
                   choices=sorted(SUPPORTED_PERSONALITIES),
                   help="NeuroBrain personality (only used when --brain neuro)")
    p.add_argument("--episodes",    type=int, default=50)
    p.add_argument("--steps",       type=int, default=500)
    p.add_argument("--fps",         type=int, default=30)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--headless",    action="store_true")
    p.add_argument("--out",         default="outputs/viz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        brain_name=args.brain,
        personality=args.personality,
        total_episodes=args.episodes,
        max_steps=args.steps,
        target_fps=args.fps,
        seed=None if args.seed == -1 else args.seed,
        headless=args.headless,
        out_dir=args.out,
    )
