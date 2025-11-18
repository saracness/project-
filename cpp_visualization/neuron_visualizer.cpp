/**
 * Real-time Neuron Learning Visualization
 *
 * Features:
 * - 3D neuron positions (projected to 2D)
 * - Dynamic synapse visualization
 * - Real-time learning graphs
 * - Activity-based coloring
 *
 * Dependencies: SFML (graphics, window, system)
 * Build: g++ -std=c++17 neuron_visualizer.cpp -lsfml-graphics -lsfml-window -lsfml-system -o neuron_visualizer
 */

#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <deque>

// JSON parsing (simple manual parsing for now)
#include <map>
#include <string>

struct Vector3 {
    float x, y, z;

    Vector3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
};

struct Neuron {
    int id;
    Vector3 position;
    float firing_rate;
    float energy;
    std::string type;
    bool is_excitatory;

    // Visual properties
    sf::Color color;
    float radius;

    Neuron() : id(0), firing_rate(0), energy(100), radius(5.0f), is_excitatory(true) {
        color = sf::Color::Blue;
    }

    void updateColor() {
        // Color based on activity
        if (is_excitatory) {
            // Blue to red based on activity
            int red = std::min(255, (int)(firing_rate * 50));
            int blue = std::max(0, 255 - red);
            color = sf::Color(red, 0, blue);
        } else {
            // Green for inhibitory
            int green = std::min(255, (int)(firing_rate * 50 + 100));
            color = sf::Color(0, green, 0);
        }

        // Fade if low energy
        if (energy < 50) {
            color.a = (sf::Uint8)(energy * 5);
        }
    }

    sf::Vector2f project2D(float width, float height, float depth) const {
        // Simple orthographic projection
        float scale_x = 800.0f / width;
        float scale_y = 600.0f / height;

        // Add depth-based offset for pseudo-3D
        float depth_offset = position.z / depth * 50.0f;

        return sf::Vector2f(
            position.x * scale_x + depth_offset,
            position.y * scale_y + depth_offset
        );
    }
};

struct Synapse {
    int pre_neuron_id;
    int post_neuron_id;
    float weight;

    Synapse(int pre, int post, float w)
        : pre_neuron_id(pre), post_neuron_id(post), weight(w) {}

    sf::Color getColor() const {
        // Color based on weight
        int alpha = std::min(255, (int)(weight * 200 + 50));
        return sf::Color(100, 100, 100, alpha);
    }

    float getThickness() const {
        return weight * 2.0f + 0.5f;
    }
};

class LearningGraph {
private:
    std::deque<float> data;
    size_t max_points;
    float min_val, max_val;
    sf::Color color;
    std::string label;

public:
    LearningGraph(const std::string& label, sf::Color color, size_t max_points = 200)
        : label(label), color(color), max_points(max_points),
          min_val(0), max_val(1) {}

    void addPoint(float value) {
        data.push_back(value);
        if (data.size() > max_points) {
            data.pop_front();
        }

        // Update min/max
        if (!data.empty()) {
            min_val = *std::min_element(data.begin(), data.end());
            max_val = *std::max_element(data.begin(), data.end());
            if (max_val == min_val) max_val = min_val + 1;
        }
    }

    void draw(sf::RenderWindow& window, sf::FloatRect bounds) {
        if (data.size() < 2) return;

        sf::VertexArray line(sf::LineStrip, data.size());

        for (size_t i = 0; i < data.size(); ++i) {
            float x = bounds.left + (i / (float)max_points) * bounds.width;
            float normalized = (data[i] - min_val) / (max_val - min_val);
            float y = bounds.top + bounds.height - (normalized * bounds.height);

            line[i].position = sf::Vector2f(x, y);
            line[i].color = color;
        }

        window.draw(line);

        // Draw bounds
        sf::RectangleShape frame(sf::Vector2f(bounds.width, bounds.height));
        frame.setPosition(bounds.left, bounds.top);
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(sf::Color(100, 100, 100));
        frame.setOutlineThickness(1);
        window.draw(frame);

        // Draw label
        // (Would need font loaded - simplified)
    }

    float getMax() const { return max_val; }
    float getMin() const { return min_val; }
};

class NeuronVisualizer {
private:
    sf::RenderWindow window;
    std::vector<Neuron> neurons;
    std::vector<Synapse> synapses;

    // Environment dimensions
    float env_width, env_height, env_depth;

    // Graphs
    LearningGraph accuracy_graph;
    LearningGraph reward_graph;
    LearningGraph synapse_graph;
    LearningGraph activity_graph;

    // Simulation state
    int timestep;
    bool paused;
    bool show_synapses;

public:
    NeuronVisualizer()
        : window(sf::VideoMode(1200, 800), "Neuron Learning Visualization"),
          accuracy_graph("Accuracy", sf::Color::Blue),
          reward_graph("Reward", sf::Color::Green),
          synapse_graph("Synapses", sf::Color::Red),
          activity_graph("Activity", sf::Color::Magenta),
          env_width(200), env_height(200), env_depth(100),
          timestep(0), paused(false), show_synapses(true)
    {
        window.setFramerateLimit(60);
    }

    void addNeuron(const Neuron& neuron) {
        neurons.push_back(neuron);
    }

    void addSynapse(const Synapse& synapse) {
        synapses.push_back(synapse);
    }

    void updateLearningMetrics(float accuracy, float reward, int num_synapses, float avg_activity) {
        accuracy_graph.addPoint(accuracy);
        reward_graph.addPoint(reward);
        synapse_graph.addPoint(num_synapses / 100.0f);  // Scale for visibility
        activity_graph.addPoint(avg_activity);
    }

    void update() {
        // Update neuron visual properties
        for (auto& neuron : neurons) {
            neuron.updateColor();
        }

        timestep++;
    }

    void handleEvents() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    paused = !paused;
                }
                if (event.key.code == sf::Keyboard::S) {
                    show_synapses = !show_synapses;
                }
            }
        }
    }

    void render() {
        window.clear(sf::Color(20, 20, 30));  // Dark background

        // Draw synapses first (behind neurons)
        if (show_synapses) {
            drawSynapses();
        }

        // Draw neurons
        drawNeurons();

        // Draw learning graphs
        drawGraphs();

        // Draw info text
        drawInfo();

        window.display();
    }

    void drawSynapses() {
        // Create ID to index map for quick lookup
        std::map<int, size_t> id_to_idx;
        for (size_t i = 0; i < neurons.size(); ++i) {
            id_to_idx[neurons[i].id] = i;
        }

        for (const auto& synapse : synapses) {
            auto pre_it = id_to_idx.find(synapse.pre_neuron_id);
            auto post_it = id_to_idx.find(synapse.post_neuron_id);

            if (pre_it == id_to_idx.end() || post_it == id_to_idx.end()) {
                continue;
            }

            const Neuron& pre = neurons[pre_it->second];
            const Neuron& post = neurons[post_it->second];

            sf::Vector2f pre_pos = pre.project2D(env_width, env_height, env_depth);
            sf::Vector2f post_pos = post.project2D(env_width, env_height, env_depth);

            // Draw line
            sf::Vertex line[] = {
                sf::Vertex(pre_pos, synapse.getColor()),
                sf::Vertex(post_pos, synapse.getColor())
            };

            window.draw(line, 2, sf::Lines);
        }
    }

    void drawNeurons() {
        for (const auto& neuron : neurons) {
            sf::Vector2f pos = neuron.project2D(env_width, env_height, env_depth);

            // Draw neuron as circle
            sf::CircleShape circle(neuron.radius);
            circle.setPosition(pos.x - neuron.radius, pos.y - neuron.radius);
            circle.setFillColor(neuron.color);
            circle.setOutlineColor(sf::Color::White);
            circle.setOutlineThickness(1);

            window.draw(circle);

            // Draw activity indicator (pulse)
            if (neuron.firing_rate > 5.0f) {
                sf::CircleShape pulse(neuron.radius + 5);
                pulse.setPosition(pos.x - neuron.radius - 5, pos.y - neuron.radius - 5);
                pulse.setFillColor(sf::Color::Transparent);
                pulse.setOutlineColor(sf::Color(255, 255, 0, 100));
                pulse.setOutlineThickness(2);
                window.draw(pulse);
            }
        }
    }

    void drawGraphs() {
        // Position graphs on right side
        float graph_width = 350;
        float graph_height = 150;
        float graph_x = 820;
        float graph_y = 50;
        float graph_spacing = 180;

        accuracy_graph.draw(window, sf::FloatRect(graph_x, graph_y, graph_width, graph_height));
        reward_graph.draw(window, sf::FloatRect(graph_x, graph_y + graph_spacing, graph_width, graph_height));
        synapse_graph.draw(window, sf::FloatRect(graph_x, graph_y + graph_spacing * 2, graph_width, graph_height));
        activity_graph.draw(window, sf::FloatRect(graph_x, graph_y + graph_spacing * 3, graph_width, graph_height));
    }

    void drawInfo() {
        // Would draw text info here if font was loaded
        // For now, just basic shapes to show info area
        sf::RectangleShape info_bg(sf::Vector2f(790, 50));
        info_bg.setPosition(10, 740);
        info_bg.setFillColor(sf::Color(40, 40, 50, 200));
        window.draw(info_bg);
    }

    bool isOpen() const {
        return window.isOpen();
    }

    bool isPaused() const {
        return paused;
    }

    void setEnvironmentDimensions(float w, float h, float d) {
        env_width = w;
        env_height = h;
        env_depth = d;
    }
};

// Simulation data loader (would load from JSON or Python pipe)
void loadSimulationData(NeuronVisualizer& viz) {
    // Example: Create some dummy neurons for testing
    for (int i = 0; i < 30; ++i) {
        Neuron n;
        n.id = i;
        n.position = Vector3(
            rand() % 200,
            rand() % 200,
            rand() % 100
        );
        n.firing_rate = (rand() % 100) / 10.0f;
        n.energy = 100 + rand() % 100;
        n.is_excitatory = (i % 5 != 0);  // 20% inhibitory
        n.type = n.is_excitatory ? "excitatory" : "inhibitory";

        viz.addNeuron(n);
    }

    // Example: Create some synapses
    for (int i = 0; i < 50; ++i) {
        int pre = rand() % 30;
        int post = rand() % 30;
        if (pre != post) {
            viz.addSynapse(Synapse(pre, post, (rand() % 100) / 100.0f));
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "Neuron Learning Visualization" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  SPACE - Pause/Resume" << std::endl;
    std::cout << "  S     - Toggle synapses" << std::endl;
    std::cout << std::endl;

    NeuronVisualizer viz;

    // Load simulation data
    if (argc > 1) {
        std::cout << "Loading data from: " << argv[1] << std::endl;
        // Would load from file here
    } else {
        std::cout << "No data file specified, using test data" << std::endl;
        loadSimulationData(viz);
    }

    viz.setEnvironmentDimensions(200, 200, 100);

    // Simulation loop
    int step = 0;
    while (viz.isOpen()) {
        viz.handleEvents();

        if (!viz.isPaused()) {
            viz.update();

            // Simulate learning progress (would come from Python)
            float accuracy = std::min(1.0f, step / 1000.0f);
            float reward = std::sin(step / 100.0f) * 0.5f + 0.5f;
            int synapses = 50 + (step / 10);
            float activity = std::abs(std::sin(step / 50.0f)) * 10;

            viz.updateLearningMetrics(accuracy, reward, synapses, activity);

            step++;
        }

        viz.render();

        // Limit update rate
        sf::sleep(sf::milliseconds(16));  // ~60 FPS
    }

    return 0;
}
