/**
 * Fast Neuron Learning Simulation with Real-Time Visualization
 *
 * Complete C++ implementation for high-performance neuron simulation
 * Features:
 * - Spatial dynamics (migration, chemotaxis)
 * - Dynamic synaptogenesis
 * - Reward-based learning
 * - Real-time SFML visualization
 * - Multi-threading ready
 * - Real AI model comparison panel (reads from ai_models_runner.py)
 *
 * Build: g++ -std=c++17 neuron_learning_fast.cpp -lsfml-graphics -lsfml-window -lsfml-system -pthread -O3
 * Run (with real AI panel):
 *   python ai_models_runner.py &
 *   ./neuron_learning_fast --neurons 100 --task xor --fps 60
 */

#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <deque>
#include <memory>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
std::normal_distribution<float> normal_dist(0.0f, 1.0f);

// Vector3D
struct Vec3 {
    float x, y, z;

    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    float distance(const Vec3& other) const {
        return (*this - other).length();
    }
};

// Synapse
struct Synapse {
    int pre_id;
    int post_id;
    float weight;
    float spine_size;
    bool is_stable;

    Synapse(int pre, int post, float w = 0.3f)
        : pre_id(pre), post_id(post), weight(w), spine_size(0.5f), is_stable(false) {}

    void update_hebbian(bool pre_active, bool post_active, float learning_rate = 0.01f) {
        if (pre_active && post_active) {
            // LTP
            weight = std::min(1.0f, weight + learning_rate);
            spine_size = std::min(1.0f, spine_size + learning_rate * 0.5f);
            is_stable = true;
        } else if (pre_active && !post_active) {
            // LTD
            weight = std::max(0.0f, weight - learning_rate * 0.5f);
            spine_size = std::max(0.1f, spine_size - learning_rate * 0.3f);
        }
    }
};

// Neuron
class Neuron {
public:
    int id;
    Vec3 position;
    Vec3 velocity;

    // Properties
    float energy;
    float firing_rate;
    float membrane_potential;
    bool is_excitatory;
    std::string neuron_type;

    // Learning
    float pre_trace;
    float post_trace;
    float reward_signal;
    float input_value;

    // Synapses
    std::vector<int> synapses_in_ids;
    std::vector<int> synapses_out_ids;

    // Visual
    sf::Color color;

    Neuron(int id, Vec3 pos, bool excitatory = true)
        : id(id), position(pos), velocity(0, 0, 0),
          energy(200.0f), firing_rate(0.0f), membrane_potential(-70.0f),
          is_excitatory(excitatory), neuron_type(excitatory ? "excitatory" : "inhibitory"),
          pre_trace(0.0f), post_trace(0.0f), reward_signal(0.0f), input_value(0.0f)
    {
        update_color();
    }

    void update(float dt) {
        // Membrane dynamics (simplified)
        float leak = (membrane_potential + 70.0f) * 0.1f;
        membrane_potential -= leak * dt;

        // Check firing
        if (membrane_potential > -55.0f) {
            fire();
            membrane_potential = -70.0f;
        }

        // Decay traces
        pre_trace *= 0.95f;
        post_trace *= 0.95f;

        // Decay firing rate
        firing_rate *= 0.98f;

        // Energy consumption
        energy -= 0.1f * dt;
        energy = std::max(0.0f, energy);

        update_color();
    }

    void fire() {
        firing_rate += 1.0f;
        post_trace = 1.0f;
    }

    void receive_input(float input) {
        input_value = input;
        if (input > 0.5f) {
            membrane_potential += 20.0f;
            pre_trace = 1.0f;
        }
    }

    void apply_reward(float reward, float learning_rate = 0.01f) {
        reward_signal = reward;
        // Reward modulates learning (will be used in synapse updates)
    }

    void update_color() {
        if (is_excitatory) {
            // Blue to red based on activity
            int red = std::min(255, (int)(firing_rate * 25));
            int blue = std::max(0, 255 - red);
            color = sf::Color(red, 0, blue, 255);
        } else {
            // Green for inhibitory
            int green = std::min(255, (int)(firing_rate * 25 + 100));
            color = sf::Color(0, green, 0, 255);
        }

        // Fade if low energy
        if (energy < 50.0f) {
            color.a = (sf::Uint8)(energy * 5);
        }
    }

    sf::Vector2f project2D(float env_width, float env_height, float env_depth) const {
        float scale_x = 800.0f / env_width;
        float scale_y = 600.0f / env_height;
        float depth_offset = (position.z / env_depth) * 30.0f;

        return sf::Vector2f(
            position.x * scale_x + depth_offset + 50,
            position.y * scale_y + depth_offset + 50
        );
    }
};

// Learning Graph
class Graph {
private:
    std::deque<float> data;
    size_t max_points;
    std::string label;
    sf::Color color;

public:
    Graph(const std::string& label, sf::Color color, size_t max_points = 200)
        : label(label), color(color), max_points(max_points) {}

    void add(float value) {
        data.push_back(value);
        if (data.size() > max_points) {
            data.pop_front();
        }
    }

    void draw(sf::RenderWindow& window, sf::FloatRect bounds) {
        if (data.size() < 2) return;

        // Find min/max
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        if (max_val == min_val) max_val = min_val + 1;

        // Draw line
        sf::VertexArray line(sf::LineStrip, data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            float x = bounds.left + (i / (float)max_points) * bounds.width;
            float normalized = (data[i] - min_val) / (max_val - min_val);
            float y = bounds.top + bounds.height - (normalized * bounds.height);

            line[i].position = sf::Vector2f(x, y);
            line[i].color = color;
        }
        window.draw(line);

        // Draw border
        sf::RectangleShape border(sf::Vector2f(bounds.width, bounds.height));
        border.setPosition(bounds.left, bounds.top);
        border.setFillColor(sf::Color::Transparent);
        border.setOutlineColor(sf::Color(80, 80, 80));
        border.setOutlineThickness(1);
        window.draw(border);
    }

    // Draw label at top-left of the graph box
    void draw_label(sf::RenderWindow& window, sf::FloatRect bounds, sf::Font& font,
                    const std::string& extra = "") {
        sf::Text text;
        text.setFont(font);
        text.setString(label + (extra.empty() ? "" : " " + extra));
        text.setCharacterSize(11);
        text.setFillColor(color);
        text.setPosition(bounds.left + 3, bounds.top + 2);
        window.draw(text);
    }
};

// ─── Real AI Models (read from Python runner) ────────────────────────────────

struct RealModelState {
    std::string name;
    float accuracy  = 0.0f;
    float avg_reward = 0.0f;
    int   episodes  = 0;
    Graph acc_graph;
    Graph rwd_graph;

    static const sf::Color MODEL_COLORS[4];

    RealModelState(const std::string& n, sf::Color col)
        : name(n),
          acc_graph("Acc", col, 150),
          rwd_graph("Rwd", sf::Color(col.r / 2, col.g / 2, col.b / 2 + 80, 255), 150) {}
};

const sf::Color RealModelState::MODEL_COLORS[4] = {
    sf::Color(255, 200,  50),   // Q-Learning  – yellow
    sf::Color( 50, 220, 255),   // Neural-Net  – cyan
    sf::Color(200,  80, 255),   // Genetic-Alg – purple
    sf::Color( 80, 255, 130),   // DQN-Simple  – green
};

class RealModelPanel {
public:
    static constexpr const char* STATE_FILE = "/tmp/real_ai_state.txt";
    static constexpr int COLOR_COUNT = 4;

    std::vector<RealModelState> models;
    int update_counter = 0;
    bool data_available = false;

    RealModelPanel() {
        // Pre-allocate with expected names & colors (will be updated from file)
        const char* names[] = {"Q-Learning", "Neural-Net", "Genetic-Alg", "DQN-Simple"};
        for (int i = 0; i < COLOR_COUNT; ++i)
            models.emplace_back(names[i], RealModelState::MODEL_COLORS[i]);
    }

    void refresh() {
        std::ifstream f(STATE_FILE);
        if (!f.is_open()) { data_available = false; return; }
        data_available = true;

        std::string line;
        int idx = 0;
        while (std::getline(f, line) && idx < (int)models.size()) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string tok;
            std::vector<std::string> parts;
            while (std::getline(ss, tok, ',')) parts.push_back(tok);
            if (parts.size() < 4) continue;

            models[idx].name       = parts[0];
            models[idx].accuracy   = std::stof(parts[1]);
            models[idx].avg_reward = std::stof(parts[2]);
            models[idx].episodes   = std::stoi(parts[3]);
            models[idx].acc_graph.add(models[idx].accuracy);
            models[idx].rwd_graph.add(std::max(0.0f, models[idx].avg_reward));
            ++idx;
        }
    }

    // Call every N frames
    void maybe_refresh(int frame, int every = 18) {
        if (frame % every == 0) refresh();
    }

    void draw(sf::RenderWindow& window, sf::Font& font, float panel_x, float panel_y,
              float panel_w, float panel_h) {
        // Panel background
        sf::RectangleShape bg(sf::Vector2f(panel_w, panel_h));
        bg.setPosition(panel_x, panel_y);
        bg.setFillColor(sf::Color(15, 15, 25));
        bg.setOutlineColor(sf::Color(60, 60, 90));
        bg.setOutlineThickness(1);
        window.draw(bg);

        // Title
        sf::Text title;
        title.setFont(font);
        title.setString(data_available ? "Real AI Models" : "Real AI Models (waiting...)");
        title.setCharacterSize(13);
        title.setFillColor(sf::Color(200, 200, 255));
        title.setStyle(sf::Text::Bold);
        title.setPosition(panel_x + 6, panel_y + 4);
        window.draw(title);

        if (!data_available) {
            sf::Text hint;
            hint.setFont(font);
            hint.setString("Run: python ai_models_runner.py");
            hint.setCharacterSize(10);
            hint.setFillColor(sf::Color(140, 140, 160));
            hint.setPosition(panel_x + 6, panel_y + 24);
            window.draw(hint);
            return;
        }

        // Each model: one row with acc-graph + rwd-graph + text stats
        float row_h   = (panel_h - 30) / (float)models.size();
        float g_w     = (panel_w - 20) * 0.45f;
        float g_h     = row_h - 18;
        float text_x  = panel_x + 12 + g_w * 2 + 8;

        for (int i = 0; i < (int)models.size(); ++i) {
            auto& m = models[i];
            float ry = panel_y + 26 + i * row_h;

            // Model name
            sf::Text nm;
            nm.setFont(font);
            nm.setString(m.name);
            nm.setCharacterSize(11);
            nm.setFillColor(RealModelState::MODEL_COLORS[i % COLOR_COUNT]);
            nm.setPosition(panel_x + 6, ry);
            window.draw(nm);

            // Accuracy bar (background)
            float bar_w = panel_w - 14;
            float bar_x = panel_x + 7;
            float bar_y = ry + 13;
            float bar_h = 6.0f;

            sf::RectangleShape bar_bg(sf::Vector2f(bar_w, bar_h));
            bar_bg.setPosition(bar_x, bar_y);
            bar_bg.setFillColor(sf::Color(40, 40, 55));
            window.draw(bar_bg);

            sf::RectangleShape bar_fill(sf::Vector2f(bar_w * m.accuracy, bar_h));
            bar_fill.setPosition(bar_x, bar_y);
            auto col = RealModelState::MODEL_COLORS[i % COLOR_COUNT];
            col.a = 200;
            bar_fill.setFillColor(col);
            window.draw(bar_fill);

            // Accuracy graph
            float gx = panel_x + 6;
            float gy = ry + 22;
            m.acc_graph.draw(window, sf::FloatRect(gx, gy, g_w, g_h));

            // Reward graph (next to it)
            m.rwd_graph.draw(window, sf::FloatRect(gx + g_w + 4, gy, g_w, g_h));

            // Stats text
            char buf[64];
            snprintf(buf, sizeof(buf), "acc=%.0f%%\nrwd=%+.2f\nep=%d",
                     m.accuracy * 100.0f, m.avg_reward, m.episodes);
            float ty = gy;
            std::istringstream lines(buf);
            std::string ln;
            while (std::getline(lines, ln)) {
                sf::Text st;
                st.setFont(font);
                st.setString(ln);
                st.setCharacterSize(9);
                st.setFillColor(sf::Color(190, 190, 190));
                st.setPosition(text_x, ty);
                window.draw(st);
                ty += 11;
            }
        }
    }
};

// Neural Network
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
    std::vector<std::unique_ptr<Synapse>> synapses;

    float env_width, env_height, env_depth;

    // Learning task
    std::vector<std::vector<float>> input_patterns;
    std::vector<std::vector<float>> target_outputs;
    std::vector<float> rewards;

    int total_trials;
    int correct_trials;
    std::deque<float> recent_accuracy;

    // Graphs
    Graph accuracy_graph;
    Graph reward_graph;
    Graph synapse_graph;
    Graph activity_graph;

    // Simulation
    float time;
    int frame_count;

    // Real AI model panel
    RealModelPanel real_panel;
    sf::Font font;
    bool font_loaded;

public:
    NeuralNetwork(float width, float height, float depth)
        : env_width(width), env_height(height), env_depth(depth),
          total_trials(0), correct_trials(0), time(0), frame_count(0),
          accuracy_graph("Accuracy", sf::Color::Blue),
          reward_graph("Reward", sf::Color::Green),
          synapse_graph("Synapses", sf::Color::Red),
          activity_graph("Activity", sf::Color::Magenta),
          font_loaded(false)
    {
        // Try to load a system font
        const char* font_paths[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            nullptr
        };
        for (int i = 0; font_paths[i]; ++i) {
            if (font.loadFromFile(font_paths[i])) {
                font_loaded = true;
                break;
            }
        }
        setup_xor_task();
    }

    void add_neuron(Vec3 position, bool excitatory = true) {
        int id = neurons.size();
        neurons.push_back(std::make_unique<Neuron>(id, position, excitatory));
    }

    void create_random_network(int num_neurons) {
        std::cout << "Creating " << num_neurons << " neurons..." << std::endl;

        for (int i = 0; i < num_neurons; ++i) {
            Vec3 pos(
                uniform_dist(gen) * env_width,
                uniform_dist(gen) * env_height,
                uniform_dist(gen) * env_depth
            );
            bool excitatory = (i % 5 != 0);  // 20% inhibitory
            add_neuron(pos, excitatory);
        }

        // Form initial synapses
        int target_synapses = num_neurons * 4;
        std::cout << "Forming " << target_synapses << " initial synapses..." << std::endl;

        for (int i = 0; i < target_synapses; ++i) {
            int pre_id = std::uniform_int_distribution<int>(0, num_neurons - 1)(gen);
            int post_id = std::uniform_int_distribution<int>(0, num_neurons - 1)(gen);

            if (pre_id != post_id) {
                connect(pre_id, post_id, uniform_dist(gen) * 0.3f + 0.2f);
            }
        }
    }

    void connect(int pre_id, int post_id, float weight = 0.3f) {
        // Check if already connected
        for (auto& syn : synapses) {
            if (syn->pre_id == pre_id && syn->post_id == post_id) {
                return;  // Already connected
            }
        }

        auto synapse = std::make_unique<Synapse>(pre_id, post_id, weight);

        if (pre_id < neurons.size() && post_id < neurons.size()) {
            neurons[pre_id]->synapses_out_ids.push_back(synapses.size());
            neurons[post_id]->synapses_in_ids.push_back(synapses.size());
        }

        synapses.push_back(std::move(synapse));
    }

    void setup_xor_task() {
        // XOR problem
        input_patterns = {
            {0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 1.0f},
            {1.0f, 1.0f, 1.0f}
        };

        target_outputs = {
            {0.0f},
            {1.0f},
            {1.0f},
            {0.0f}
        };

        rewards = {1.0f, 1.0f, 1.0f, 1.0f};
    }

    void update(float dt) {
        time += dt;
        frame_count++;

        // Update all neurons
        for (auto& neuron : neurons) {
            neuron->update(dt);

            // Add energy (simulate metabolism)
            neuron->energy = std::min(200.0f, neuron->energy + 0.5f * dt);
        }

        // Dynamic synaptogenesis (every 10 frames)
        if (frame_count % 10 == 0) {
            attempt_synapse_formation();
        }

        // Run learning trial (every 5 frames)
        if (frame_count % 5 == 0) {
            run_learning_trial();
        }

        // Update graphs (every 10 frames)
        if (frame_count % 10 == 0) {
            float acc = recent_accuracy.empty() ? 0.0f :
                        std::accumulate(recent_accuracy.begin(), recent_accuracy.end(), 0.0f) / recent_accuracy.size();
            float avg_reward = 0.0f;
            float avg_firing = 0.0f;

            for (auto& n : neurons) {
                avg_reward += n->reward_signal;
                avg_firing += n->firing_rate;
            }
            avg_reward /= neurons.size();
            avg_firing /= neurons.size();

            accuracy_graph.add(acc);
            reward_graph.add(avg_reward);
            synapse_graph.add(synapses.size() / 100.0f);
            activity_graph.add(avg_firing);
        }

        // Refresh real AI model state from file (every 18 frames)
        real_panel.maybe_refresh(frame_count);
    }

    void attempt_synapse_formation() {
        // Try to form new synapse between nearby neurons
        if (uniform_dist(gen) > 0.1f) return;  // 10% chance

        int pre_id = std::uniform_int_distribution<int>(0, neurons.size() - 1)(gen);

        // Find nearby neurons
        std::vector<int> candidates;
        for (size_t i = 0; i < neurons.size(); ++i) {
            if (i == pre_id) continue;

            float dist = neurons[pre_id]->position.distance(neurons[i]->position);
            if (dist < 50.0f) {  // Within 50 units
                candidates.push_back(i);
            }
        }

        if (!candidates.empty()) {
            int post_id = candidates[std::uniform_int_distribution<int>(0, candidates.size() - 1)(gen)];
            connect(pre_id, post_id);
        }
    }

    void run_learning_trial() {
        if (input_patterns.empty()) return;

        // Select random pattern
        int pattern_idx = std::uniform_int_distribution<int>(0, input_patterns.size() - 1)(gen);
        auto& input = input_patterns[pattern_idx];
        auto& target = target_outputs[pattern_idx];
        float reward = rewards[pattern_idx];

        // Present input to first neurons
        size_t num_inputs = std::min(input.size(), neurons.size());
        for (size_t i = 0; i < num_inputs; ++i) {
            neurons[i]->receive_input(input[i]);
        }

        // Read output from last neurons
        std::vector<float> output;
        size_t output_start = neurons.size() > 5 ? neurons.size() - 5 : 0;
        for (size_t i = output_start; i < neurons.size(); ++i) {
            output.push_back(neurons[i]->firing_rate > 5.0f ? 1.0f : 0.0f);
        }

        // Check if correct
        bool correct = true;
        for (size_t i = 0; i < std::min(target.size(), output.size()); ++i) {
            if (std::abs(output[i] - target[i]) > 0.5f) {
                correct = false;
                break;
            }
        }

        float actual_reward = correct ? reward : -0.1f;
        if (correct) correct_trials++;
        total_trials++;

        recent_accuracy.push_back(correct ? 1.0f : 0.0f);
        if (recent_accuracy.size() > 100) recent_accuracy.pop_front();

        // Apply reward to all neurons and synapses
        for (auto& neuron : neurons) {
            neuron->apply_reward(actual_reward);
        }

        for (auto& synapse : synapses) {
            if (synapse->pre_id < neurons.size() && synapse->post_id < neurons.size()) {
                bool pre_active = neurons[synapse->pre_id]->pre_trace > 0.1f;
                bool post_active = neurons[synapse->post_id]->post_trace > 0.1f;

                float modulated_lr = 0.01f * (1.0f + actual_reward);
                synapse->update_hebbian(pre_active, post_active, modulated_lr);
            }
        }
    }

    void render(sf::RenderWindow& window) {
        window.clear(sf::Color(20, 20, 30));

        // Draw synapses
        for (auto& synapse : synapses) {
            if (synapse->pre_id >= neurons.size() || synapse->post_id >= neurons.size()) continue;

            auto pre_pos = neurons[synapse->pre_id]->project2D(env_width, env_height, env_depth);
            auto post_pos = neurons[synapse->post_id]->project2D(env_width, env_height, env_depth);

            int alpha = std::min(255, (int)(synapse->weight * 150 + 30));
            sf::Color syn_color(100, 100, 100, alpha);

            sf::Vertex line[] = {
                sf::Vertex(pre_pos, syn_color),
                sf::Vertex(post_pos, syn_color)
            };
            window.draw(line, 2, sf::Lines);
        }

        // Draw neurons
        for (auto& neuron : neurons) {
            auto pos = neuron->project2D(env_width, env_height, env_depth);

            sf::CircleShape circle(5);
            circle.setPosition(pos.x - 5, pos.y - 5);
            circle.setFillColor(neuron->color);
            circle.setOutlineColor(sf::Color::White);
            circle.setOutlineThickness(1);
            window.draw(circle);

            // Activity pulse
            if (neuron->firing_rate > 5.0f) {
                sf::CircleShape pulse(10);
                pulse.setPosition(pos.x - 10, pos.y - 10);
                pulse.setFillColor(sf::Color::Transparent);
                pulse.setOutlineColor(sf::Color(255, 255, 0, 100));
                pulse.setOutlineThickness(2);
                window.draw(pulse);
            }
        }

        // ── Biological simulation graphs (middle column 850-1200) ──
        float graph_x = 855;
        float graph_y = 50;
        float graph_w = 330;
        float graph_h = 140;
        float spacing = 165;

        auto draw_graph_labeled = [&](Graph& g, float gx, float gy, float gw, float gh) {
            g.draw(window, sf::FloatRect(gx, gy, gw, gh));
            if (font_loaded)
                g.draw_label(window, sf::FloatRect(gx, gy, gw, gh), font);
        };

        draw_graph_labeled(accuracy_graph, graph_x, graph_y,                graph_w, graph_h);
        draw_graph_labeled(reward_graph,   graph_x, graph_y + spacing,      graph_w, graph_h);
        draw_graph_labeled(synapse_graph,  graph_x, graph_y + spacing * 2,  graph_w, graph_h);
        draw_graph_labeled(activity_graph, graph_x, graph_y + spacing * 3,  graph_w, graph_h);

        // ── Real AI Models panel (right column 1210-1590) ──
        if (font_loaded)
            real_panel.draw(window, font, 1210, 10, 380, 780);

        window.display();
    }

    void print_stats() {
        float acc = total_trials > 0 ? (correct_trials / (float)total_trials) * 100.0f : 0.0f;
        float recent_acc = recent_accuracy.empty() ? 0.0f :
            (std::accumulate(recent_accuracy.begin(), recent_accuracy.end(), 0.0f) / recent_accuracy.size()) * 100.0f;

        std::cout << "Frame " << frame_count
                  << " | Neurons: " << neurons.size()
                  << " | Synapses: " << synapses.size()
                  << " | Trials: " << total_trials
                  << " | Acc: " << acc << "%"
                  << " | Recent: " << recent_acc << "%"
                  << std::endl;
    }
};

// Main
int main(int argc, char** argv) {
    // Parse arguments
    int num_neurons = 60;
    int target_fps = 60;
    std::string task = "xor";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--neurons" && i + 1 < argc) {
            num_neurons = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            target_fps = std::stoi(argv[++i]);
        } else if (arg == "--task" && i + 1 < argc) {
            task = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --neurons N   Number of neurons (default: 60)\n"
                      << "  --fps N       Target FPS (default: 60)\n"
                      << "  --task NAME   Task name (default: xor)\n"
                      << "  --help        Show this help\n";
            return 0;
        }
    }

    std::cout << "Brain Neuron Learning Simulation\n";
    std::cout << "====================================\n";
    std::cout << "Neurons: " << num_neurons << "\n";
    std::cout << "Target FPS: " << target_fps << "\n";
    std::cout << "Task: " << task << "\n";
    std::cout << "Real AI panel: run 'python ai_models_runner.py' for live model comparison\n";
    std::cout << "====================================\n";

    // Create window (wider to fit Real AI Models panel on the right)
    sf::RenderWindow window(sf::VideoMode(1600, 800),
                            "Neuron Learning + Real AI Models - Fast C++ Simulation");
    window.setFramerateLimit(target_fps);

    // Create network
    NeuralNetwork network(300, 300, 150);
    network.create_random_network(num_neurons);

    std::cout << "\n🚀 Simulation started! Press ESC to exit.\n";
    std::cout << "Controls: SPACE=Pause, S=Stats\n\n";

    // Simulation loop
    sf::Clock clock;
    bool paused = false;
    int stats_frame = 0;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                } else if (event.key.code == sf::Keyboard::Space) {
                    paused = !paused;
                } else if (event.key.code == sf::Keyboard::S) {
                    network.print_stats();
                }
            }
        }

        if (!paused) {
            float dt = clock.restart().asSeconds();
            dt = std::min(dt, 0.1f);  // Clamp for stability

            network.update(dt);

            // Print stats every 300 frames
            if (++stats_frame % 300 == 0) {
                network.print_stats();
            }
        }

        network.render(window);
    }

    network.print_stats();
    std::cout << "\n✅ Simulation ended.\n";

    return 0;
}
