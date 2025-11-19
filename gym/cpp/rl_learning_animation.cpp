/*
 * RL LEARNING ANIMATION - C++ Version
 * ===================================
 *
 * Real-time visualization of Q-Learning algorithm learning to solve GridWorld.
 *
 * WHAT YOU'LL SEE:
 * - Agent exploring the grid (red circle)
 * - Q-values updating in real-time (color-coded)
 * - Policy arrows forming (best action per state)
 * - Learning curves (episode rewards, lengths)
 * - Statistics (epsilon decay, Q-value growth)
 *
 * CONTROLS:
 * - SPACE: Pause/Resume
 * - R: Reset learning
 * - +/-: Speed up/slow down
 * - S: Skip to next episode
 * - Q: Quit
 *
 * Compile:
 *   g++ -std=c++17 -O3 rl_learning_animation.cpp -lsfml-graphics -lsfml-window -lsfml-system -o rl_learning_animation
 *
 * Run:
 *   ./rl_learning_animation
 */

#include <SFML/Graphics.hpp>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <deque>

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const int GRID_SIZE = 5;
const int CELL_SIZE = 100;
const int STATS_WIDTH = 400;
const int WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + STATS_WIDTH;
const int WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE;

const float LEARNING_RATE = 0.1f;
const float DISCOUNT_FACTOR = 0.99f;
const float INITIAL_EPSILON = 0.3f;
const float EPSILON_DECAY = 0.995f;
const float EPSILON_MIN = 0.01f;

const float GOAL_REWARD = 10.0f;
const float WALL_PENALTY = -1.0f;
const float STEP_PENALTY = -0.1f;

const int MAX_STEPS_PER_EPISODE = 100;

// ============================================================================
// GRID WORLD ENVIRONMENT
// ============================================================================

enum Action { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };

struct State {
    int row, col;

    bool operator==(const State& other) const {
        return row == other.row && col == other.col;
    }

    int toIndex() const {
        return row * GRID_SIZE + col;
    }
};

class GridWorld {
public:
    State agent_pos;
    State start_pos{0, 0};
    State goal_pos{GRID_SIZE-1, GRID_SIZE-1};

    std::vector<State> walls;

    int steps;
    float total_reward;
    bool done;

    GridWorld() {
        // Add some walls for interesting navigation
        if (GRID_SIZE >= 5) {
            walls.push_back({2, 1});
            walls.push_back({2, 2});
            walls.push_back({2, 3});
        }

        reset();
    }

    void reset() {
        agent_pos = start_pos;
        steps = 0;
        total_reward = 0;
        done = false;
    }

    std::pair<State, float> step(Action action) {
        if (done) return {agent_pos, 0.0f};

        // Calculate new position
        State new_pos = agent_pos;

        switch(action) {
            case UP:    new_pos.row--; break;
            case DOWN:  new_pos.row++; break;
            case LEFT:  new_pos.col--; break;
            case RIGHT: new_pos.col++; break;
        }

        // Check validity
        float reward = STEP_PENALTY;

        if (!isValid(new_pos)) {
            // Hit wall or boundary
            reward = WALL_PENALTY;
        } else {
            // Valid move
            agent_pos = new_pos;

            // Check goal
            if (agent_pos == goal_pos) {
                reward = GOAL_REWARD;
                done = true;
            }
        }

        steps++;
        total_reward += reward;

        if (steps >= MAX_STEPS_PER_EPISODE) {
            done = true;
        }

        return {agent_pos, reward};
    }

    bool isValid(const State& s) const {
        // Check boundaries
        if (s.row < 0 || s.row >= GRID_SIZE || s.col < 0 || s.col >= GRID_SIZE) {
            return false;
        }

        // Check walls
        for (const auto& wall : walls) {
            if (s == wall) return false;
        }

        return true;
    }
};

// ============================================================================
// Q-LEARNING AGENT
// ============================================================================

class QLearningAgent {
public:
    static constexpr int N_STATES = GRID_SIZE * GRID_SIZE;
    static constexpr int N_ACTIONS = 4;

    std::array<std::array<float, N_ACTIONS>, N_STATES> Q;

    float alpha;   // Learning rate
    float gamma;   // Discount factor
    float epsilon; // Exploration rate

    // Statistics
    std::deque<float> episode_rewards;
    std::deque<int> episode_lengths;
    std::deque<float> avg_q_values;

    int episode_count;

    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;

    QLearningAgent()
        : alpha(LEARNING_RATE),
          gamma(DISCOUNT_FACTOR),
          epsilon(INITIAL_EPSILON),
          episode_count(0),
          gen(std::random_device{}()),
          uniform_dist(0.0f, 1.0f)
    {
        // Initialize Q-table to zeros
        for (auto& state_q : Q) {
            state_q.fill(0.0f);
        }
    }

    Action selectAction(const State& state, bool training = true) {
        // ε-greedy policy
        if (training && uniform_dist(gen) < epsilon) {
            // Explore: random action
            return static_cast<Action>(std::rand() % N_ACTIONS);
        } else {
            // Exploit: best action
            return getBestAction(state);
        }
    }

    void learn(const State& state, Action action, float reward,
               const State& next_state, bool done) {
        int s = state.toIndex();
        int s_next = next_state.toIndex();

        // Current Q-value
        float current_q = Q[s][action];

        // TD target
        float target;
        if (done) {
            target = reward;
        } else {
            // Bellman optimality: r + γ max_a' Q(s',a')
            float max_q_next = *std::max_element(Q[s_next].begin(), Q[s_next].end());
            target = reward + gamma * max_q_next;
        }

        // Q-update: Q(s,a) ← Q(s,a) + α[target - Q(s,a)]
        Q[s][action] += alpha * (target - current_q);
    }

    Action getBestAction(const State& state) const {
        int s = state.toIndex();

        auto max_it = std::max_element(Q[s].begin(), Q[s].end());
        return static_cast<Action>(std::distance(Q[s].begin(), max_it));
    }

    float getMaxQ(const State& state) const {
        int s = state.toIndex();
        return *std::max_element(Q[s].begin(), Q[s].end());
    }

    float getQ(const State& state, Action action) const {
        return Q[state.toIndex()][action];
    }

    void decayEpsilon() {
        epsilon = std::max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    void saveStatistics(float episode_reward, int episode_length) {
        episode_rewards.push_back(episode_reward);
        episode_lengths.push_back(episode_length);

        // Calculate average Q-value
        float avg_q = 0;
        for (const auto& state_q : Q) {
            avg_q += *std::max_element(state_q.begin(), state_q.end());
        }
        avg_q /= N_STATES;
        avg_q_values.push_back(avg_q);

        // Keep only last 100 episodes
        if (episode_rewards.size() > 100) {
            episode_rewards.pop_front();
            episode_lengths.pop_front();
            avg_q_values.pop_front();
        }

        episode_count++;
    }
};

// ============================================================================
// VISUALIZATION
// ============================================================================

class Visualizer {
public:
    sf::RenderWindow& window;
    sf::Font font;

    GridWorld& env;
    QLearningAgent& agent;

    // Colors
    sf::Color grid_color{50, 50, 50};
    sf::Color wall_color{80, 80, 80};
    sf::Color goal_color{255, 215, 0};  // Gold
    sf::Color agent_color{255, 50, 50}; // Red
    sf::Color text_color{255, 255, 255};

    Visualizer(sf::RenderWindow& win, GridWorld& e, QLearningAgent& a)
        : window(win), env(e), agent(a)
    {
        // Try to load font (optional)
        font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    }

    void render(int current_step_in_episode, bool show_policy = true) {
        window.clear(sf::Color(20, 20, 30));

        // Draw grid
        drawGrid();

        // Draw Q-values (heatmap)
        if (show_policy) {
            drawQValueHeatmap();
        }

        // Draw walls
        drawWalls();

        // Draw goal
        drawGoal();

        // Draw policy arrows
        if (show_policy && agent.episode_count > 5) {
            drawPolicyArrows();
        }

        // Draw agent
        drawAgent();

        // Draw statistics panel
        drawStatsPanel(current_step_in_episode);

        window.display();
    }

private:
    void drawGrid() {
        for (int i = 0; i <= GRID_SIZE; i++) {
            // Vertical lines
            sf::RectangleShape line(sf::Vector2f(2, GRID_SIZE * CELL_SIZE));
            line.setPosition(i * CELL_SIZE, 0);
            line.setFillColor(grid_color);
            window.draw(line);

            // Horizontal lines
            line.setSize(sf::Vector2f(GRID_SIZE * CELL_SIZE, 2));
            line.setPosition(0, i * CELL_SIZE);
            window.draw(line);
        }
    }

    void drawQValueHeatmap() {
        // Find min and max Q-values for normalization
        float min_q = 0;
        float max_q = 0;

        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                State s{r, c};
                if (env.isValid(s) && !(s == env.goal_pos)) {
                    float q = agent.getMaxQ(s);
                    min_q = std::min(min_q, q);
                    max_q = std::max(max_q, q);
                }
            }
        }

        float range = max_q - min_q;
        if (range < 0.01f) range = 1.0f;

        // Draw colored cells based on Q-values
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                State s{r, c};

                if (!env.isValid(s) || s == env.goal_pos) continue;

                float q = agent.getMaxQ(s);
                float normalized = (q - min_q) / range;

                // Color: blue (low) → green (medium) → red (high)
                sf::Color color;
                if (normalized < 0.5f) {
                    // Blue to green
                    color = sf::Color(0, static_cast<sf::Uint8>(normalized * 2 * 255),
                                     static_cast<sf::Uint8>((1 - normalized * 2) * 255), 80);
                } else {
                    // Green to red
                    color = sf::Color(static_cast<sf::Uint8>((normalized - 0.5f) * 2 * 255),
                                     static_cast<sf::Uint8>((1 - (normalized - 0.5f) * 2) * 255),
                                     0, 80);
                }

                sf::RectangleShape cell(sf::Vector2f(CELL_SIZE - 4, CELL_SIZE - 4));
                cell.setPosition(c * CELL_SIZE + 2, r * CELL_SIZE + 2);
                cell.setFillColor(color);
                window.draw(cell);
            }
        }
    }

    void drawWalls() {
        for (const auto& wall : env.walls) {
            sf::RectangleShape rect(sf::Vector2f(CELL_SIZE - 4, CELL_SIZE - 4));
            rect.setPosition(wall.col * CELL_SIZE + 2, wall.row * CELL_SIZE + 2);
            rect.setFillColor(wall_color);
            window.draw(rect);
        }
    }

    void drawGoal() {
        sf::RectangleShape rect(sf::Vector2f(CELL_SIZE - 10, CELL_SIZE - 10));
        rect.setPosition(env.goal_pos.col * CELL_SIZE + 5, env.goal_pos.row * CELL_SIZE + 5);
        rect.setFillColor(goal_color);
        window.draw(rect);

        // "G" text
        sf::Text text("G", font, 60);
        text.setFillColor(sf::Color::Black);
        text.setStyle(sf::Text::Bold);
        sf::FloatRect bounds = text.getLocalBounds();
        text.setPosition(
            env.goal_pos.col * CELL_SIZE + CELL_SIZE/2 - bounds.width/2,
            env.goal_pos.row * CELL_SIZE + CELL_SIZE/2 - bounds.height/2 - 10
        );
        window.draw(text);
    }

    void drawPolicyArrows() {
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                State s{r, c};

                if (!env.isValid(s) || s == env.goal_pos) continue;

                Action best = agent.getBestAction(s);

                float cx = c * CELL_SIZE + CELL_SIZE / 2;
                float cy = r * CELL_SIZE + CELL_SIZE / 2;

                // Arrow direction
                float dx = 0, dy = 0;
                switch(best) {
                    case UP:    dy = -30; break;
                    case DOWN:  dy = 30; break;
                    case LEFT:  dx = -30; break;
                    case RIGHT: dx = 30; break;
                }

                // Draw arrow
                sf::ConvexShape arrow(7);
                arrow.setPoint(0, sf::Vector2f(cx, cy));
                arrow.setPoint(1, sf::Vector2f(cx + dx * 0.7f, cy + dy * 0.7f));

                // Arrowhead
                float angle = std::atan2(dy, dx);
                float head_size = 10;
                arrow.setPoint(2, sf::Vector2f(cx + dx, cy + dy));
                arrow.setPoint(3, sf::Vector2f(
                    cx + dx - head_size * std::cos(angle - 0.5f),
                    cy + dy - head_size * std::sin(angle - 0.5f)
                ));
                arrow.setPoint(4, sf::Vector2f(cx + dx, cy + dy));
                arrow.setPoint(5, sf::Vector2f(
                    cx + dx - head_size * std::cos(angle + 0.5f),
                    cy + dy - head_size * std::sin(angle + 0.5f)
                ));
                arrow.setPoint(6, sf::Vector2f(cx + dx, cy + dy));

                arrow.setFillColor(sf::Color(100, 100, 255, 200));
                window.draw(arrow);
            }
        }
    }

    void drawAgent() {
        sf::CircleShape circle(CELL_SIZE / 3);
        circle.setPosition(
            env.agent_pos.col * CELL_SIZE + CELL_SIZE/2 - CELL_SIZE/3,
            env.agent_pos.row * CELL_SIZE + CELL_SIZE/2 - CELL_SIZE/3
        );
        circle.setFillColor(agent_color);
        circle.setOutlineColor(sf::Color::White);
        circle.setOutlineThickness(3);
        window.draw(circle);
    }

    void drawStatsPanel(int current_step) {
        int panel_x = GRID_SIZE * CELL_SIZE;

        // Background
        sf::RectangleShape bg(sf::Vector2f(STATS_WIDTH, WINDOW_HEIGHT));
        bg.setPosition(panel_x, 0);
        bg.setFillColor(sf::Color(30, 30, 40));
        window.draw(bg);

        // Title
        drawText("Q-LEARNING", panel_x + 20, 20, 24, sf::Color::Yellow);
        drawText("LIVE TRAINING", panel_x + 20, 50, 18, sf::Color::White);

        // Episode info
        drawText("Episode: " + std::to_string(agent.episode_count), panel_x + 20, 100, 16);
        drawText("Step: " + std::to_string(current_step) + " / " + std::to_string(env.steps),
                panel_x + 20, 130, 16);
        drawText("Reward: " + formatFloat(env.total_reward, 2), panel_x + 20, 160, 16);

        // Agent parameters
        drawText("PARAMETERS:", panel_x + 20, 210, 16, sf::Color::Cyan);
        drawText("Alpha: " + formatFloat(agent.alpha, 3), panel_x + 20, 240, 14);
        drawText("Gamma: " + formatFloat(agent.gamma, 3), panel_x + 20, 265, 14);
        drawText("Epsilon: " + formatFloat(agent.epsilon, 3), panel_x + 20, 290, 14);

        // Statistics
        if (!agent.episode_rewards.empty()) {
            float avg_reward = 0;
            for (float r : agent.episode_rewards) avg_reward += r;
            avg_reward /= agent.episode_rewards.size();

            float avg_length = 0;
            for (int l : agent.episode_lengths) avg_length += l;
            avg_length /= agent.episode_lengths.size();

            drawText("STATISTICS (last " + std::to_string(agent.episode_rewards.size()) + "):",
                    panel_x + 20, 340, 16, sf::Color::Cyan);
            drawText("Avg Reward: " + formatFloat(avg_reward, 2), panel_x + 20, 370, 14);
            drawText("Avg Length: " + formatFloat(avg_length, 1), panel_x + 20, 395, 14);

            if (!agent.avg_q_values.empty()) {
                drawText("Avg Q-value: " + formatFloat(agent.avg_q_values.back(), 2),
                        panel_x + 20, 420, 14);
            }
        }

        // Mini learning curve
        if (agent.episode_rewards.size() > 2) {
            drawMiniGraph(panel_x + 20, 470, STATS_WIDTH - 40, 80,
                         agent.episode_rewards, "Reward");
        }

        // Controls
        drawText("CONTROLS:", panel_x + 20, WINDOW_HEIGHT - 120, 14, sf::Color::Green);
        drawText("SPACE: Pause", panel_x + 20, WINDOW_HEIGHT - 95, 12);
        drawText("+/-: Speed", panel_x + 20, WINDOW_HEIGHT - 75, 12);
        drawText("S: Skip episode", panel_x + 20, WINDOW_HEIGHT - 55, 12);
        drawText("R: Reset", panel_x + 20, WINDOW_HEIGHT - 35, 12);
    }

    void drawText(const std::string& str, float x, float y, int size,
                  sf::Color color = sf::Color::White) {
        sf::Text text(str, font, size);
        text.setPosition(x, y);
        text.setFillColor(color);
        window.draw(text);
    }

    std::string formatFloat(float value, int precision) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << value;
        return oss.str();
    }

    void drawMiniGraph(float x, float y, float w, float h,
                      const std::deque<float>& data, const std::string& title) {
        // Background
        sf::RectangleShape bg(sf::Vector2f(w, h));
        bg.setPosition(x, y);
        bg.setFillColor(sf::Color(50, 50, 60));
        bg.setOutlineColor(sf::Color(100, 100, 120));
        bg.setOutlineThickness(1);
        window.draw(bg);

        // Title
        sf::Text text(title, font, 12);
        text.setPosition(x + 5, y - 20);
        text.setFillColor(sf::Color::White);
        window.draw(text);

        if (data.size() < 2) return;

        // Find min/max
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        float range = max_val - min_val;
        if (range < 0.01f) range = 1.0f;

        // Draw line
        sf::VertexArray line(sf::LineStrip, data.size());

        for (size_t i = 0; i < data.size(); i++) {
            float px = x + (i / float(data.size() - 1)) * w;
            float py = y + h - ((data[i] - min_val) / range) * h;

            line[i].position = sf::Vector2f(px, py);
            line[i].color = sf::Color::Green;
        }

        window.draw(line);
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  Q-LEARNING ANIMATION - Live Training Visualization\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "Watch the agent learn to navigate from (0,0) to (4,4)!\n";
    std::cout << "\n";
    std::cout << "WHAT YOU'LL SEE:\n";
    std::cout << "  - Red circle: Agent exploring\n";
    std::cout << "  - Color heatmap: Q-values (blue=low, red=high)\n";
    std::cout << "  - Blue arrows: Learned policy\n";
    std::cout << "  - Gold square: Goal\n";
    std::cout << "  - Stats panel: Real-time learning metrics\n";
    std::cout << "\n";
    std::cout << "CONTROLS:\n";
    std::cout << "  SPACE: Pause/Resume\n";
    std::cout << "  +/-: Speed up/slow down\n";
    std::cout << "  S: Skip to next episode\n";
    std::cout << "  R: Reset learning\n";
    std::cout << "  Q/ESC: Quit\n";
    std::cout << "\n";
    std::cout << "Starting in 2 seconds...\n";
    std::cout << "============================================================\n";

    sf::sleep(sf::seconds(2));

    // Create window
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                           "Q-Learning Live Animation - 120 FPS");
    window.setFramerateLimit(120);

    // Create environment and agent
    GridWorld env;
    QLearningAgent agent;
    Visualizer viz(window, env, agent);

    // Animation state
    bool paused = false;
    float animation_speed = 1.0f;  // Steps per frame
    int frame_counter = 0;
    bool in_episode = false;
    int step_in_episode = 0;

    sf::Clock clock;

    std::cout << "\n✓ Window opened! Training started...\n\n";

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape || event.key.code == sf::Keyboard::Q) {
                    window.close();
                }
                else if (event.key.code == sf::Keyboard::Space) {
                    paused = !paused;
                    std::cout << (paused ? "⏸  PAUSED" : "▶  RESUMED") << "\n";
                }
                else if (event.key.code == sf::Keyboard::Equal || event.key.code == sf::Keyboard::Add) {
                    animation_speed *= 1.5f;
                    std::cout << "⏩ Speed: " << animation_speed << "x\n";
                }
                else if (event.key.code == sf::Keyboard::Hyphen || event.key.code == sf::Keyboard::Subtract) {
                    animation_speed = std::max(0.1f, animation_speed / 1.5f);
                    std::cout << "⏪ Speed: " << animation_speed << "x\n";
                }
                else if (event.key.code == sf::Keyboard::S) {
                    // Skip to next episode
                    env.done = true;
                    std::cout << "⏭  Skipping to next episode...\n";
                }
                else if (event.key.code == sf::Keyboard::R) {
                    // Reset learning
                    agent = QLearningAgent();
                    env.reset();
                    in_episode = false;
                    std::cout << "🔄 Learning reset!\n";
                }
            }
        }

        // Update (training step)
        if (!paused) {
            for (int i = 0; i < static_cast<int>(animation_speed); i++) {
                if (!in_episode) {
                    // Start new episode
                    env.reset();
                    in_episode = true;
                    step_in_episode = 0;

                    if (agent.episode_count % 10 == 0 && agent.episode_count > 0) {
                        float avg_reward = 0;
                        for (float r : agent.episode_rewards) avg_reward += r;
                        avg_reward /= agent.episode_rewards.size();

                        std::cout << "Episode " << agent.episode_count
                                  << ": Avg Reward = " << std::fixed << std::setprecision(2)
                                  << avg_reward
                                  << ", ε = " << std::setprecision(3) << agent.epsilon << "\n";
                    }
                }

                if (!env.done) {
                    // Take action
                    Action action = agent.selectAction(env.agent_pos);
                    auto [next_state, reward] = env.step(action);

                    // Learn
                    agent.learn(env.agent_pos, action, reward, next_state, env.done);

                    step_in_episode++;

                    // Slow down visualization for individual steps
                    if (animation_speed < 2.0f) {
                        break;  // Only one step per frame for slow speed
                    }
                } else {
                    // Episode done
                    agent.saveStatistics(env.total_reward, env.steps);
                    agent.decayEpsilon();
                    in_episode = false;
                }
            }
        }

        // Render
        viz.render(step_in_episode, agent.episode_count > 0);

        frame_counter++;
    }

    // Print final statistics
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  TRAINING COMPLETE!\n";
    std::cout << "============================================================\n";
    std::cout << "\n";
    std::cout << "Total episodes: " << agent.episode_count << "\n";

    if (!agent.episode_rewards.empty()) {
        float avg_reward = 0;
        for (float r : agent.episode_rewards) avg_reward += r;
        avg_reward /= agent.episode_rewards.size();

        std::cout << "Final avg reward (last " << agent.episode_rewards.size()
                  << " episodes): " << avg_reward << "\n";
    }

    std::cout << "Final epsilon: " << agent.epsilon << "\n";
    std::cout << "\n";
    std::cout << "✓ Agent learned to navigate GridWorld!\n";
    std::cout << "\n";

    return 0;
}
