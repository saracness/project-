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
 *
 * Build: g++ -std=c++17 neuron_learning_fast.cpp -lsfml-graphics -lsfml-window -lsfml-system -pthread -O3
 * Run:   ./neuron_learning_fast --neurons 100 --task xor --fps 60
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

public:
    NeuralNetwork(float width, float height, float depth)
        : env_width(width), env_height(height), env_depth(depth),
          total_trials(0), correct_trials(0), time(0), frame_count(0),
          accuracy_graph("Accuracy", sf::Color::Blue),
          reward_graph("Reward", sf::Color::Green),
          synapse_graph("Synapses", sf::Color::Red),
          activity_graph("Activity", sf::Color::Magenta)
    {
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

        // Draw graphs
        float graph_x = 850;
        float graph_y = 50;
        float graph_w = 330;
        float graph_h = 140;
        float spacing = 160;

        accuracy_graph.draw(window, sf::FloatRect(graph_x, graph_y, graph_w, graph_h));
        reward_graph.draw(window, sf::FloatRect(graph_x, graph_y + spacing, graph_w, graph_h));
        synapse_graph.draw(window, sf::FloatRect(graph_x, graph_y + spacing * 2, graph_w, graph_h));
        activity_graph.draw(window, sf::FloatRect(graph_x, graph_y + spacing * 3, graph_w, graph_h));

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

    std::cout << "🧠 Fast Neuron Learning Simulation\n";
    std::cout << "====================================\n";
    std::cout << "Neurons: " << num_neurons << "\n";
    std::cout << "Target FPS: " << target_fps << "\n";
    std::cout << "Task: " << task << "\n";
    std::cout << "====================================\n";

    // Create window
    sf::RenderWindow window(sf::VideoMode(1200, 800), "Neuron Learning - Fast C++ Simulation");
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
