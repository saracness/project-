"""
Live visualizer — every pixel you see is driven by the RL brain.

Each frame:
  1. brain.decide_action(state)   → picks action (+ stores Q-values)
  2. env.step(action_idx)         → organism moves, eats, survives
  3. pygame draws the result      → you see what the brain decided

Nothing is pre-scripted. The organism moves because the DQN said so.

Controls
--------
  SPACE       pause / resume
  RIGHT       one step while paused
  +/-         speed up / slow down
  R           restart episode
  Q           quit and save summary plot

Usage
-----
  python scripts/train_visual.py --brain dqn --fps 30 --episodes 50
  python scripts/train_visual.py --brain dqn --headless --episodes 200 --out outputs/viz
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.gym_env import MicrolifeEnv, ACTIONS, N_ACTIONS
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SIM_W, SIM_H    = 600, 600   # simulation canvas
PANEL_W         = 320        # right-side info panel
WIN_W           = SIM_W + PANEL_W
WIN_H           = SIM_H

SCALE           = SIM_W / 300  # world (300×300) → screen pixels

# Colours
BG              = (15,  20,  30)
GRID            = (25,  35,  50)
FOOD_COL        = (80, 220,  80)
ORG_COL         = (60, 180, 255)
TRAIL_COL       = (40, 100, 160)
TEMP_HOT        = (200,  60,  40, 60)
TEMP_COLD       = (60,  120, 220, 60)
PANEL_BG        = (20,  25,  38)
TEXT_COL        = (200, 210, 230)
DIM_COL         = (90, 100, 120)
BAR_POS         = (80, 180, 80)
BAR_NEG         = (220, 80,  60)
ARROW_COL       = (255, 230,  50)
BEST_ARROW      = (255, 100, 255)


# ---------------------------------------------------------------------------
# Brain factory (same as train.py)
# ---------------------------------------------------------------------------
def make_brain(name: str):
    name = name.lower()
    if name == "qlearning":
        return QLearningBrain(learning_rate=0.1, discount_factor=0.95, epsilon=0.5)
    if name == "dqn":
        return DQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    if name == "ddqn":
        return DoubleDQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    raise ValueError(f"Unknown brain '{name}'")


# ---------------------------------------------------------------------------
# Q-value extraction — works for both Q-table and DQN
# ---------------------------------------------------------------------------
def get_q_values(brain, state: dict) -> np.ndarray:
    """Return raw Q-values for current state (9 actions)."""
    if isinstance(brain, DQNBrain):
        vec = brain.get_state_vector(state)
        return brain._forward(vec)
    elif isinstance(brain, QLearningBrain):
        discrete = brain._discretize_state(state)
        if discrete in brain.q_table:
            return brain.q_table[discrete].copy()
        return np.zeros(N_ACTIONS)
    return np.zeros(N_ACTIONS)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def world_to_screen(wx, wy):
    return int(wx * SCALE), int(wy * SCALE)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_grid(surf):
    for x in range(0, SIM_W, 50):
        pygame.draw.line(surf, GRID, (x, 0), (x, SIM_H))
    for y in range(0, SIM_H, 50):
        pygame.draw.line(surf, GRID, (0, y), (SIM_W, y))


def draw_temp_zones(surf, env):
    zone_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    for zone in env._env.temperature_zones:
        r = int(zone.radius * SCALE)
        cx, cy = world_to_screen(zone.x, zone.y)
        col = TEMP_HOT if zone.temperature > 0 else TEMP_COLD
        pygame.draw.circle(zone_surf, col, (cx, cy), r)
    surf.blit(zone_surf, (0, 0))


def draw_food(surf, env):
    for food in env._env.food_particles:
        if not food.consumed:
            fx, fy = world_to_screen(food.x, food.y)
            pygame.draw.circle(surf, FOOD_COL, (fx, fy), max(3, int(2 * SCALE)))


def draw_organism(surf, env, q_values: np.ndarray):
    o = env._organism
    if not o.alive:
        return

    # Trail
    trail = o.trail[-30:]
    for i in range(1, len(trail)):
        alpha = int(255 * i / len(trail))
        col = (*TRAIL_COL, alpha)
        p1 = world_to_screen(*trail[i-1])
        p2 = world_to_screen(*trail[i])
        pygame.draw.line(surf, TRAIL_COL, p1, p2, 1)

    ox, oy = world_to_screen(o.x, o.y)
    radius = max(6, int(o.size * SCALE))

    # Energy ring (green → red as energy depletes)
    energy_ratio = max(0.0, min(1.0, o.energy / 200.0))
    ring_col = (
        int(255 * (1 - energy_ratio)),
        int(255 * energy_ratio),
        80,
    )
    pygame.draw.circle(surf, ring_col, (ox, oy), radius + 3, 2)

    # Body
    pygame.draw.circle(surf, ORG_COL, (ox, oy), radius)

    # Draw all action arrows (faint), best action bold
    best_action = int(np.argmax(q_values))
    q_min, q_max = q_values.min(), q_values.max()
    q_range = q_max - q_min if q_max != q_min else 1.0

    for i, (dx, dy) in enumerate(ACTIONS[:-1]):  # skip Stay
        norm = (q_values[i] - q_min) / q_range  # 0..1
        length = int(norm * 28 + 4)
        angle = math.atan2(-dy, dx)  # pygame y-axis flipped
        ex = ox + int(math.cos(angle) * length)
        ey = oy + int(math.sin(angle) * length)
        col = BEST_ARROW if i == best_action else (*DIM_COL, 80)
        width = 2 if i == best_action else 1
        pygame.draw.line(surf, col if i == best_action else DIM_COL,
                         (ox, oy), (ex, ey), width)

    # Arrow head for best action
    if best_action < 8:
        dx, dy = ACTIONS[best_action]
        angle = math.atan2(-dy, dx)
        tip_len = 24
        ex = ox + int(math.cos(angle) * tip_len)
        ey = oy + int(math.sin(angle) * tip_len)
        pygame.draw.circle(surf, BEST_ARROW, (ex, ey), 3)


def draw_panel(surf, font_lg, font_sm, brain, env, ep, total_ep,
               ep_reward, step, q_values: np.ndarray, action_idx: int,
               fps: float, paused: bool):
    """Draw right-side info panel."""
    surf.fill(PANEL_BG)
    y = 14

    def txt(text, f=None, col=TEXT_COL):
        nonlocal y
        label = (f or font_sm).render(text, True, col)
        surf.blit(label, (12, y))
        y += label.get_height() + 4

    # Header
    txt(f"MicrolifeEnv  —  {type(brain).__name__}", font_lg)
    txt(f"Episode {ep} / {total_ep}", col=DIM_COL)
    if paused:
        txt("⏸  PAUSED  (SPACE to resume)", col=(255, 200, 50))
    else:
        txt(f"FPS: {fps:.0f}   Step: {step}", col=DIM_COL)

    y += 8
    # Organism stats
    o = env._organism
    txt("─── Organism ───────────────", col=DIM_COL)
    energy_pct = int(o.energy / 200 * 100) if o.alive else 0
    txt(f"Energy : {o.energy:6.1f}  ({energy_pct}%)")
    txt(f"Age    : {o.age:6d}")
    txt(f"Alive  : {'YES' if o.alive else 'NO'}")
    txt(f"Reward : {ep_reward:+.2f}")

    y += 8
    # Brain stats
    epsilon = getattr(brain, "epsilon", float("nan"))
    txt("─── Brain ───────────────────", col=DIM_COL)
    txt(f"Epsilon      : {epsilon:.4f}")
    txt(f"Decisions    : {brain.decision_count}")
    txt(f"Total reward : {brain.total_reward:.1f}")
    if hasattr(brain, "memory"):
        txt(f"Replay buffer: {len(brain.memory)}")

    y += 8
    # Q-values bar chart
    txt("─── Q-values (this state) ──", col=DIM_COL)
    ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Stay"]
    bar_max_w = PANEL_W - 80
    q_min, q_max = q_values.min(), q_values.max()
    q_range = q_max - q_min if q_max != q_min else 1.0
    best = int(np.argmax(q_values))

    for i, (name, qv) in enumerate(zip(ACTION_NAMES, q_values)):
        bar_w = int((qv - q_min) / q_range * bar_max_w)
        col = BEST_ARROW if i == best else (
            BAR_POS if qv >= 0 else BAR_NEG
        )
        label = font_sm.render(f"{name:>4}", True, TEXT_COL if i == best else DIM_COL)
        surf.blit(label, (12, y))
        pygame.draw.rect(surf, col, (52, y + 2, max(1, bar_w), 10))
        val_lbl = font_sm.render(f"{qv:+.3f}", True, TEXT_COL if i == best else DIM_COL)
        surf.blit(val_lbl, (52 + bar_max_w + 4, y))
        y += 16

    y += 4
    # Chosen action highlight
    txt(f"→ Chose: {ACTION_NAMES[action_idx]}", col=ARROW_COL)

    y += 8
    # Controls hint
    txt("─── Controls ────────────────", col=DIM_COL)
    for line in ["SPACE  pause/resume", "+/-  speed", "R  restart ep", "Q  quit"]:
        txt(line, col=DIM_COL)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    brain_name: str = "dqn",
    total_episodes: int = 50,
    max_steps: int = 500,
    target_fps: int = 30,
    seed: int | None = 42,
    headless: bool = False,
    out_dir: str = "outputs/viz",
):
    os.makedirs(out_dir, exist_ok=True)

    env = MicrolifeEnv(width=300, height=300, n_food=20,
                       n_temp_zones=3, max_steps=max_steps, seed=seed)
    brain = make_brain(brain_name)

    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")

    pygame.init()
    win = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"MicrolifeEnv — {type(brain).__name__}")
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm = pygame.font.SysFont("monospace", 12)

    sim_surf   = pygame.Surface((SIM_W, SIM_H))
    panel_surf = pygame.Surface((PANEL_W, WIN_H))

    paused     = False
    speed      = 1          # steps per frame
    ep_rewards = []

    for ep in range(1, total_episodes + 1):
        obs      = env.reset()
        state    = env.obs_to_brain_state(obs)
        ep_rew   = 0.0
        step     = 0
        q_values = np.zeros(N_ACTIONS)
        action_idx = 8  # Stay
        info     = {}

        if headless:
            # ── Headless: run at full speed, no FPS cap, no event loop ──
            while True:
                if env._step_count >= max_steps or not env._organism.alive:
                    break

                action_dict = brain.decide_action(state)
                action_idx  = action_dict["_action_idx"]
                q_values    = get_q_values(brain, state)

                obs_next, reward, done, info = env.step(action_idx)
                next_state = env.obs_to_brain_state(obs_next)
                brain.learn(state, action_dict, reward, next_state, done)

                ep_rew += reward
                step   += 1
                state   = next_state

                if done:
                    break

            # Save screenshot once per episode in headless mode
            _render(win, sim_surf, panel_surf, font_lg, font_sm,
                    env, brain, ep, total_episodes, ep_rew,
                    step, q_values, action_idx, 0.0, paused=False)
            if ep % 10 == 0 or ep == 1:
                path = os.path.join(out_dir, f"ep_{ep:04d}.png")
                pygame.image.save(win, path)
                print(f"Ep {ep:4d}/{total_episodes} | "
                      f"reward={ep_rew:7.2f} | "
                      f"energy={info.get('energy',0):6.1f} | "
                      f"ε={getattr(brain,'epsilon',0):.3f} | "
                      f"saved {path}")

        else:
            # ── Interactive: event loop + FPS cap ───────────────────────
            episode_done = False
            while not episode_done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return ep_rewards
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                            speed = min(speed + 1, 20)
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            speed = max(speed - 1, 1)
                        elif event.key == pygame.K_r:
                            episode_done = True
                        elif event.key == pygame.K_q:
                            _save_summary(ep_rewards, out_dir)
                            pygame.quit()
                            return ep_rewards

                if paused:
                    _render(win, sim_surf, panel_surf, font_lg, font_sm,
                            env, brain, ep, total_episodes, ep_rew,
                            step, q_values, action_idx, clock.get_fps(), paused=True)
                    clock.tick(30)
                    continue

                for _ in range(speed):
                    if env._step_count >= max_steps or not env._organism.alive:
                        episode_done = True
                        break

                    action_dict = brain.decide_action(state)
                    action_idx  = action_dict["_action_idx"]
                    q_values    = get_q_values(brain, state)

                    obs_next, reward, done, info = env.step(action_idx)
                    next_state = env.obs_to_brain_state(obs_next)
                    brain.learn(state, action_dict, reward, next_state, done)

                    ep_rew += reward
                    step   += 1
                    state   = next_state

                    if done:
                        episode_done = True
                        break

                _render(win, sim_surf, panel_surf, font_lg, font_sm,
                        env, brain, ep, total_episodes, ep_rew,
                        step, q_values, action_idx, clock.get_fps(), paused=False)
                clock.tick(target_fps)

        ep_rewards.append(ep_rew)

        # Save screenshot every 10 episodes
        if ep % 10 == 0 or ep == 1:
            path = os.path.join(out_dir, f"ep_{ep:04d}.png")
            pygame.image.save(win, path)
            print(f"Ep {ep:4d}/{total_episodes} | "
                  f"reward={ep_rew:7.2f} | "
                  f"energy={info.get('energy',0):6.1f} | "
                  f"ε={getattr(brain,'epsilon',0):.3f} | "
                  f"saved {path}")

    _save_summary(ep_rewards, out_dir)
    pygame.quit()
    return ep_rewards


def _render(win, sim_surf, panel_surf, font_lg, font_sm,
            env, brain, ep, total_ep, ep_rew, step,
            q_values, action_idx, fps, paused):
    """Compose and blit one frame."""
    sim_surf.fill(BG)
    draw_grid(sim_surf)
    draw_temp_zones(sim_surf, env)
    draw_food(sim_surf, env)
    draw_organism(sim_surf, env, q_values)

    draw_panel(panel_surf, font_lg, font_sm, brain, env,
               ep, total_ep, ep_rew, step,
               q_values, action_idx, fps, paused)

    win.blit(sim_surf, (0, 0))
    win.blit(panel_surf, (SIM_W, 0))
    pygame.display.flip()


def _save_summary(ep_rewards: list, out_dir: str):
    """Save reward-over-episodes matplotlib plot."""
    if not ep_rewards:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ep_rewards, alpha=0.5, color="steelblue", label="Episode reward")

        # Rolling average
        window = min(20, len(ep_rewards))
        if len(ep_rewards) >= window:
            roll = np.convolve(ep_rewards,
                               np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(ep_rewards)), roll,
                    color="orange", linewidth=2,
                    label=f"{window}-ep rolling mean")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("MicrolifeEnv — RL agent learning curve")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(out_dir, "learning_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Learning curve saved → {path}")
    except Exception as e:
        print(f"Could not save summary plot: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Watch a real RL agent learn in MicrolifeEnv")
    p.add_argument("--brain",    default="dqn",
                   choices=["qlearning", "dqn", "ddqn"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--steps",    type=int, default=500)
    p.add_argument("--fps",      type=int, default=30)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--headless", action="store_true",
                   help="No window — saves screenshots only")
    p.add_argument("--out",      default="outputs/viz",
                   help="Output directory for screenshots + learning curve")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = None if args.seed == -1 else args.seed
    run(
        brain_name=args.brain,
        total_episodes=args.episodes,
        max_steps=args.steps,
        target_fps=args.fps,
        seed=seed,
        headless=args.headless,
        out_dir=args.out,
    )
