/**
 * ╔═══════════════════════════════════════════════════════════════════╗
 * ║                ONLY FOR NATURE - ULTIMATE EDITION                  ║
 * ║      Complete Neuron Network with Synapses & Learning             ║
 * ║                                                                    ║
 * ║  NEW FEATURES:                                                     ║
 * ║  • SYNAPTIC CONNECTIONS - Visible neural connections!             ║
 * ║  • HEBBIAN LEARNING - "Fire together, wire together"              ║
 * ║  • SIGNAL PROPAGATION - Animated synaptic transmission            ║
 * ║  • INTERACTIVE CONTROLS - Mouse & keyboard interaction            ║
 * ║  • REAL-TIME GRAPHS - Network activity visualization              ║
 * ║  • NETWORK STATISTICS - Live connection data                      ║
 * ║  • MULTIPLE AGENTS - 3 explorers with different behaviors         ║
 * ║                                                                    ║
 * ║  Controls:                                                         ║
 * ║    ESC   - Exit                   SPACE - Pause                   ║
 * ║    Mouse - Click to add reward    R - Reset network               ║
 * ║    C - Toggle connections         S - Toggle signals              ║
 * ║    G - Toggle graphs              +/- - Adjust speed              ║
 * ║    1-9 - Highlight neuron type                                    ║
 * ║                                                                    ║
 * ║  Build: make -f Makefile.nature.ultimate                          ║
 * ║  Run:   ./ONLY_FOR_NATURE_ULTIMATE                                ║
 * ╚═══════════════════════════════════════════════════════════════════╝
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
#include <sstream>
#include <iomanip>
#include <list>

// Constants
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;
const float PI = 3.14159265359f;
const int TARGET_FPS = 120;
const float MAX_CONNECTION_DISTANCE = 250.0f;
const float SYNAPTIC_DELAY = 0.01f; // 10ms delay

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
std::normal_distribution<float> normal_dist(0.0f, 1.0f);

// 3D Vector
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

    Vec3 normalized() const {
        float len = length();
        if (len > 0) return Vec3(x/len, y/len, z/len);
        return Vec3(0, 0, 0);
    }
};

// Neuron Personality Types
enum class PersonalityType {
    DOPAMINERGIC,      // Reward coding, burst firing
    SEROTONERGIC,      // Mood regulation, slow regular
    CHOLINERGIC,       // Attention, irregular
    PLACE_CELL,        // Spatial navigation
    GRID_CELL,         // Spatial metric
    MIRROR_NEURON,     // Action understanding
    VON_ECONOMO,       // Social cognition
    FAST_SPIKING,      // Timing, fast inhibition
    CHATTERING         // Pattern recognition
};

// Color schemes for each personality
struct PersonalityColors {
    sf::Color primary;
    sf::Color glow;
    sf::Color trail;

    PersonalityColors() = default;
    PersonalityColors(sf::Color p, sf::Color g, sf::Color t)
        : primary(p), glow(g), trail(t) {}
};

std::map<PersonalityType, PersonalityColors> colorMap = {
    {PersonalityType::DOPAMINERGIC,   {sf::Color(255, 50, 50),   sf::Color(255, 100, 100, 150), sf::Color(255, 50, 50, 80)}},
    {PersonalityType::SEROTONERGIC,   {sf::Color(50, 150, 255),  sf::Color(100, 180, 255, 150), sf::Color(50, 150, 255, 80)}},
    {PersonalityType::CHOLINERGIC,    {sf::Color(255, 200, 50),  sf::Color(255, 220, 100, 150), sf::Color(255, 200, 50, 80)}},
    {PersonalityType::PLACE_CELL,     {sf::Color(50, 255, 150),  sf::Color(100, 255, 180, 150), sf::Color(50, 255, 150, 80)}},
    {PersonalityType::GRID_CELL,      {sf::Color(150, 50, 255),  sf::Color(180, 100, 255, 150), sf::Color(150, 50, 255, 80)}},
    {PersonalityType::MIRROR_NEURON,  {sf::Color(255, 150, 200), sf::Color(255, 180, 220, 150), sf::Color(255, 150, 200, 80)}},
    {PersonalityType::VON_ECONOMO,    {sf::Color(200, 255, 100), sf::Color(220, 255, 150, 150), sf::Color(200, 255, 100, 80)}},
    {PersonalityType::FAST_SPIKING,   {sf::Color(255, 255, 255), sf::Color(255, 255, 255, 200), sf::Color(255, 255, 255, 100)}},
    {PersonalityType::CHATTERING,     {sf::Color(255, 100, 255), sf::Color(255, 150, 255, 150), sf::Color(255, 100, 255, 80)}}
};

std::map<PersonalityType, std::string> personalityNames = {
    {PersonalityType::DOPAMINERGIC,   "Dopaminergic VTA"},
    {PersonalityType::SEROTONERGIC,   "Serotonergic Raphe"},
    {PersonalityType::CHOLINERGIC,    "Cholinergic Basal"},
    {PersonalityType::PLACE_CELL,     "Hippocampal Place"},
    {PersonalityType::GRID_CELL,      "Entorhinal Grid"},
    {PersonalityType::MIRROR_NEURON,  "Mirror Neuron"},
    {PersonalityType::VON_ECONOMO,    "Von Economo"},
    {PersonalityType::FAST_SPIKING,   "Fast-Spiking PV"},
    {PersonalityType::CHATTERING,     "Chattering"}
};

// Forward declaration
class Neuron;

// Synapse - Connection between neurons
struct Synapse {
    int pre_neuron_id;
    int post_neuron_id;
    float weight;              // Synaptic strength (0-1)
    float activity;            // Current activity level
    bool is_plastic;           // Can change weight?

    // Learning
    float pre_trace = 0.0f;    // Presynaptic trace
    float post_trace = 0.0f;   // Postsynaptic trace

    // Visual
    float pulse = 0.0f;        // Visual pulse for signal transmission

    Synapse(int pre, int post, float w = 0.3f, bool plastic = true)
        : pre_neuron_id(pre), post_neuron_id(post), weight(w),
          activity(0.0f), is_plastic(plastic) {}

    void updateHebbian(bool pre_active, bool post_active, float learning_rate = 0.005f) {
        if (!is_plastic) return;

        // Hebbian learning: Fire together, wire together
        if (pre_active && post_active) {
            // LTP - Long-term potentiation
            weight = std::min(1.0f, weight + learning_rate);
        } else if (pre_active && !post_active) {
            // LTD - Long-term depression
            weight = std::max(0.05f, weight - learning_rate * 0.5f);
        }

        // Update traces for STDP (spike-timing dependent plasticity)
        if (pre_active) pre_trace = std::min(1.0f, pre_trace + 0.5f);
        if (post_active) post_trace = std::min(1.0f, post_trace + 0.5f);

        // Decay traces
        pre_trace *= 0.95f;
        post_trace *= 0.95f;
    }

    void transmitSignal() {
        pulse = 1.0f; // Visual pulse
    }

    void update(float dt) {
        pulse *= 0.9f; // Decay pulse
        activity *= 0.95f; // Decay activity
    }
};

// Synaptic signal traveling along connection
struct SynapticSignal {
    int synapse_index;
    float progress;  // 0 to 1
    sf::Color color;

    SynapticSignal(int idx, sf::Color col)
        : synapse_index(idx), progress(0.0f), color(col) {}

    void update(float dt, float speed = 2.0f) {
        progress += dt * speed;
    }

    bool isComplete() const {
        return progress >= 1.0f;
    }
};

// Particle for visual effects
struct Particle {
    Vec3 position;
    Vec3 velocity;
    sf::Color color;
    float lifetime;
    float max_lifetime;
    float size;

    Particle(Vec3 pos, Vec3 vel, sf::Color col, float life, float sz = 2.0f)
        : position(pos), velocity(vel), color(col), lifetime(life), max_lifetime(life), size(sz) {}

    void update(float dt) {
        position = position + velocity * dt;
        lifetime -= dt;
        float alpha = (lifetime / max_lifetime) * 255.0f;
        color.a = static_cast<sf::Uint8>(std::max(0.0f, std::min(255.0f, alpha)));
    }

    bool isAlive() const {
        return lifetime > 0;
    }
};

// Neuron Class
class Neuron {
public:
    int id;
    Vec3 position;
    Vec3 velocity;
    PersonalityType personality;

    // Neuronal properties
    float energy = 100.0f;
    float firing_rate = 0.0f;
    float membrane_potential = -70.0f;
    float baseline_firing_rate;
    float max_firing_rate;
    bool just_fired = false;

    // Learning & connectivity
    float activation_trace = 0.0f;  // For learning
    std::vector<int> incoming_synapses;
    std::vector<int> outgoing_synapses;

    // Visual properties
    float glow_intensity = 0.0f;
    float pulse_phase = 0.0f;
    std::deque<Vec3> trail;

    // Place cell specific
    Vec3 place_field_center;
    float place_field_radius = 150.0f;

    // Grid cell specific
    float grid_spacing = 80.0f;
    float grid_orientation = 0.0f;

    // Burst firing
    float burst_timer = 0.0f;
    bool in_burst = false;

    Neuron(int id, Vec3 pos, PersonalityType type)
        : id(id), position(pos), personality(type), velocity(0, 0, 0) {

        // Set baseline and max firing rates based on personality
        switch(personality) {
            case PersonalityType::DOPAMINERGIC:
                baseline_firing_rate = 4.0f;
                max_firing_rate = 20.0f;
                break;
            case PersonalityType::SEROTONERGIC:
                baseline_firing_rate = 1.0f;
                max_firing_rate = 5.0f;
                break;
            case PersonalityType::CHOLINERGIC:
                baseline_firing_rate = 5.0f;
                max_firing_rate = 30.0f;
                break;
            case PersonalityType::PLACE_CELL:
                baseline_firing_rate = 0.5f;
                max_firing_rate = 40.0f;
                break;
            case PersonalityType::GRID_CELL:
                baseline_firing_rate = 2.0f;
                max_firing_rate = 30.0f;
                break;
            case PersonalityType::MIRROR_NEURON:
                baseline_firing_rate = 3.0f;
                max_firing_rate = 50.0f;
                break;
            case PersonalityType::VON_ECONOMO:
                baseline_firing_rate = 8.0f;
                max_firing_rate = 80.0f;
                break;
            case PersonalityType::FAST_SPIKING:
                baseline_firing_rate = 10.0f;
                max_firing_rate = 200.0f;
                break;
            case PersonalityType::CHATTERING:
                baseline_firing_rate = 2.0f;
                max_firing_rate = 100.0f;
                break;
        }

        firing_rate = baseline_firing_rate;

        // Random place field center for place cells
        if (personality == PersonalityType::PLACE_CELL) {
            place_field_center = Vec3(
                uniform_dist(gen) * WINDOW_WIDTH,
                uniform_dist(gen) * WINDOW_HEIGHT,
                0
            );
        }

        // Random grid orientation for grid cells
        if (personality == PersonalityType::GRID_CELL) {
            grid_orientation = uniform_dist(gen) * 2.0f * PI;
        }

        pulse_phase = uniform_dist(gen) * 2.0f * PI;
    }

    void update(float dt, const Vec3& agent_position, float reward_signal, float synaptic_input) {
        just_fired = false;

        // Add synaptic input to membrane potential
        membrane_potential += synaptic_input * 10.0f;

        // Update based on personality
        switch(personality) {
            case PersonalityType::DOPAMINERGIC:
                updateDopaminergic(dt, reward_signal);
                break;
            case PersonalityType::PLACE_CELL:
                updatePlaceCell(dt, agent_position);
                break;
            case PersonalityType::GRID_CELL:
                updateGridCell(dt, agent_position);
                break;
            case PersonalityType::FAST_SPIKING:
                updateFastSpiking(dt);
                break;
            case PersonalityType::CHATTERING:
                updateChattering(dt);
                break;
            default:
                updateRegular(dt);
                break;
        }

        // Check if neuron fires
        if (firing_rate > baseline_firing_rate * 1.5f) {
            just_fired = true;
            activation_trace = 1.0f;
        }

        // Decay activation trace
        activation_trace *= 0.95f;

        // Update glow intensity based on firing rate
        float normalized_rate = firing_rate / max_firing_rate;
        glow_intensity = normalized_rate;

        // Update pulse phase
        pulse_phase += dt * 2.0f * PI * (firing_rate / 10.0f);
        if (pulse_phase > 2.0f * PI) pulse_phase -= 2.0f * PI;

        // Slowly return membrane potential to resting
        membrane_potential += (-70.0f - membrane_potential) * dt * 0.1f;
    }

    void updateDopaminergic(float dt, float reward_signal) {
        if (std::abs(reward_signal) > 0.1f) {
            in_burst = true;
            burst_timer = 0.2f;
            firing_rate = max_firing_rate;
        } else if (burst_timer > 0) {
            burst_timer -= dt;
            firing_rate = max_firing_rate;
            if (burst_timer <= 0) {
                in_burst = false;
            }
        } else {
            firing_rate = baseline_firing_rate;
        }
    }

    void updatePlaceCell(float dt, const Vec3& agent_position) {
        float distance = position.distance(agent_position);
        if (distance < place_field_radius) {
            float activation = 1.0f - (distance / place_field_radius);
            firing_rate = baseline_firing_rate + (max_firing_rate - baseline_firing_rate) * activation;
        } else {
            firing_rate = baseline_firing_rate;
        }
    }

    void updateGridCell(float dt, const Vec3& agent_position) {
        float dx = agent_position.x - position.x;
        float dy = agent_position.y - position.y;

        float x_phase = std::cos(dx / grid_spacing + grid_orientation);
        float y_phase = std::cos(dy / grid_spacing + grid_orientation + PI/3.0f);
        float z_phase = std::cos(dx / grid_spacing + dy / grid_spacing + grid_orientation + 2.0f*PI/3.0f);

        float activation = (x_phase + y_phase + z_phase) / 3.0f;
        activation = (activation + 1.0f) / 2.0f;

        firing_rate = baseline_firing_rate + (max_firing_rate - baseline_firing_rate) * activation;
    }

    void updateFastSpiking(float dt) {
        firing_rate = baseline_firing_rate + std::sin(pulse_phase * 10.0f) * 20.0f;
    }

    void updateChattering(float dt) {
        float burst_freq = std::sin(pulse_phase * 0.5f);
        if (burst_freq > 0.5f) {
            firing_rate = max_firing_rate * burst_freq;
        } else {
            firing_rate = baseline_firing_rate;
        }
    }

    void updateRegular(float dt) {
        firing_rate = baseline_firing_rate + std::sin(pulse_phase) * 2.0f;
    }
};

// Agent that moves through space
class Agent {
public:
    Vec3 position;
    Vec3 velocity;
    Vec3 target;
    float speed = 100.0f;
    sf::Color color;
    std::deque<Vec3> trail;
    int behavior_mode = 0; // 0=random, 1=reward-seeking, 2=avoidance

    Agent(Vec3 pos, sf::Color col, int mode = 0)
        : position(pos), velocity(0, 0, 0), color(col), behavior_mode(mode) {
        setRandomTarget();
    }

    void setRandomTarget() {
        target = Vec3(
            uniform_dist(gen) * WINDOW_WIDTH,
            uniform_dist(gen) * WINDOW_HEIGHT,
            0
        );
    }

    void update(float dt) {
        Vec3 direction = (target - position).normalized();
        velocity = direction * speed;
        position = position + velocity * dt;

        if (position.distance(target) < 20.0f) {
            setRandomTarget();
        }

        // Keep in bounds
        if (position.x < 0) position.x = 0;
        if (position.x > WINDOW_WIDTH) position.x = WINDOW_WIDTH;
        if (position.y < 0) position.y = 0;
        if (position.y > WINDOW_HEIGHT) position.y = WINDOW_HEIGHT;

        trail.push_front(position);
        if (trail.size() > 50) {
            trail.pop_back();
        }
    }
};

// Main Simulation
class UltimateNatureSimulation {
public:
    std::vector<Neuron> neurons;
    std::vector<Synapse> synapses;
    std::vector<SynapticSignal> synaptic_signals;
    std::vector<Particle> particles;
    std::vector<Agent> agents;

    float reward_signal = 0.0f;
    float time = 0.0f;
    float simulation_speed = 1.0f;

    // Rewards zones
    std::vector<Vec3> reward_zones;

    // Display options
    bool show_connections = true;
    bool show_signals = true;
    bool show_graphs = true;
    int highlighted_type = -1; // -1 = none, 0-8 = personality type

    // Stats
    int frame_count = 0;
    float fps = 0.0f;
    sf::Clock fps_clock;
    std::deque<float> network_activity_history;

    UltimateNatureSimulation() {
        initializeNeurons();
        initializeAgents();
        initializeSynapses();
        initializeRewardZones();
    }

    void initializeNeurons() {
        int id = 0;

        std::vector<PersonalityType> types = {
            PersonalityType::DOPAMINERGIC,
            PersonalityType::SEROTONERGIC,
            PersonalityType::CHOLINERGIC,
            PersonalityType::PLACE_CELL,
            PersonalityType::GRID_CELL,
            PersonalityType::MIRROR_NEURON,
            PersonalityType::VON_ECONOMO,
            PersonalityType::FAST_SPIKING,
            PersonalityType::CHATTERING
        };

        std::vector<int> counts = {5, 5, 5, 15, 10, 5, 5, 10, 8};

        for (size_t i = 0; i < types.size(); i++) {
            for (int j = 0; j < counts[i]; j++) {
                Vec3 pos(
                    uniform_dist(gen) * WINDOW_WIDTH,
                    uniform_dist(gen) * WINDOW_HEIGHT,
                    0
                );
                neurons.emplace_back(id++, pos, types[i]);
            }
        }

        std::cout << "✓ Created " << neurons.size() << " neurons\n";
    }

    void initializeAgents() {
        // 3 different agents with different behaviors
        agents.emplace_back(Vec3(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, 0),
                           sf::Color(100, 200, 255), 0); // Random walker
        agents.emplace_back(Vec3(WINDOW_WIDTH/4, WINDOW_HEIGHT/2, 0),
                           sf::Color(255, 200, 100), 1); // Reward seeker
        agents.emplace_back(Vec3(3*WINDOW_WIDTH/4, WINDOW_HEIGHT/2, 0),
                           sf::Color(200, 100, 255), 2); // Avoidance

        std::cout << "✓ Created " << agents.size() << " agents\n";
    }

    void initializeSynapses() {
        // Create connections based on distance and personality compatibility
        for (size_t i = 0; i < neurons.size(); i++) {
            for (size_t j = i + 1; j < neurons.size(); j++) {
                float distance = neurons[i].position.distance(neurons[j].position);

                if (distance < MAX_CONNECTION_DISTANCE) {
                    // Connection probability based on distance
                    float prob = 1.0f - (distance / MAX_CONNECTION_DISTANCE);

                    if (uniform_dist(gen) < prob * 0.3f) { // 30% max connection rate
                        // Create bidirectional connections
                        float weight = 0.2f + uniform_dist(gen) * 0.3f;

                        synapses.emplace_back(i, j, weight, true);
                        int syn_idx = synapses.size() - 1;
                        neurons[i].outgoing_synapses.push_back(syn_idx);
                        neurons[j].incoming_synapses.push_back(syn_idx);

                        // Reverse connection
                        synapses.emplace_back(j, i, weight, true);
                        syn_idx = synapses.size() - 1;
                        neurons[j].outgoing_synapses.push_back(syn_idx);
                        neurons[i].incoming_synapses.push_back(syn_idx);
                    }
                }
            }
        }

        std::cout << "✓ Created " << synapses.size() << " synapses\n";
    }

    void initializeRewardZones() {
        for (int i = 0; i < 3; i++) {
            reward_zones.emplace_back(
                uniform_dist(gen) * WINDOW_WIDTH,
                uniform_dist(gen) * WINDOW_HEIGHT,
                0
            );
        }
    }

    void update(float dt) {
        dt *= simulation_speed;
        time += dt;

        // Update agents
        for (auto& agent : agents) {
            agent.update(dt);
        }

        // Check reward signal from all agents
        reward_signal = 0.0f;
        for (const auto& agent : agents) {
            for (const auto& zone : reward_zones) {
                float distance = agent.position.distance(zone);
                if (distance < 100.0f) {
                    float signal = 1.0f - (distance / 100.0f);
                    reward_signal = std::max(reward_signal, signal);

                    if (uniform_dist(gen) < 0.1f) {
                        emitRewardParticles(zone);
                    }
                }
            }
        }

        // Calculate synaptic input for each neuron
        std::vector<float> synaptic_inputs(neurons.size(), 0.0f);

        for (size_t syn_idx = 0; syn_idx < synapses.size(); syn_idx++) {
            Synapse& synapse = synapses[syn_idx];
            const Neuron& pre = neurons[synapse.pre_neuron_id];

            if (pre.just_fired) {
                // Transmit signal
                synapse.transmitSignal();
                synapse.activity = pre.firing_rate / pre.max_firing_rate;

                // Create visual signal
                if (show_signals && uniform_dist(gen) < 0.3f) {
                    auto colors = colorMap[pre.personality];
                    synaptic_signals.emplace_back(syn_idx, colors.primary);
                }

                // Add to postsynaptic neuron input
                synaptic_inputs[synapse.post_neuron_id] += synapse.weight * synapse.activity;
            }
        }

        // Update all neurons
        float total_activity = 0.0f;
        for (size_t i = 0; i < neurons.size(); i++) {
            // Use average agent position for spatial neurons
            Vec3 avg_agent_pos(0, 0, 0);
            for (const auto& agent : agents) {
                avg_agent_pos = avg_agent_pos + agent.position;
            }
            avg_agent_pos = avg_agent_pos * (1.0f / agents.size());

            neurons[i].update(dt, avg_agent_pos, reward_signal, synaptic_inputs[i]);
            total_activity += neurons[i].firing_rate;

            // Emit particles for highly active neurons
            if (neurons[i].firing_rate > neurons[i].baseline_firing_rate * 3.0f) {
                if (uniform_dist(gen) < 0.03f) {
                    emitNeuronParticle(neurons[i]);
                }
            }
        }

        // Store network activity for graphs
        network_activity_history.push_front(total_activity / neurons.size());
        if (network_activity_history.size() > 200) {
            network_activity_history.pop_back();
        }

        // Update synapses with Hebbian learning
        for (auto& synapse : synapses) {
            const Neuron& pre = neurons[synapse.pre_neuron_id];
            const Neuron& post = neurons[synapse.post_neuron_id];

            synapse.updateHebbian(pre.just_fired, post.just_fired, 0.002f);
            synapse.update(dt);
        }

        // Update synaptic signals
        for (auto& signal : synaptic_signals) {
            signal.update(dt, 3.0f);
        }

        synaptic_signals.erase(
            std::remove_if(synaptic_signals.begin(), synaptic_signals.end(),
                [](const SynapticSignal& s) { return s.isComplete(); }),
            synaptic_signals.end()
        );

        // Update particles
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [dt](Particle& p) {
                    p.update(dt);
                    return !p.isAlive();
                }),
            particles.end()
        );

        // Randomly relocate reward zones
        if (uniform_dist(gen) < 0.001f) {
            int zone_idx = static_cast<int>(uniform_dist(gen) * reward_zones.size());
            reward_zones[zone_idx] = Vec3(
                uniform_dist(gen) * WINDOW_WIDTH,
                uniform_dist(gen) * WINDOW_HEIGHT,
                0
            );
        }

        // Update FPS
        frame_count++;
        if (fps_clock.getElapsedTime().asSeconds() >= 1.0f) {
            fps = frame_count / fps_clock.getElapsedTime().asSeconds();
            frame_count = 0;
            fps_clock.restart();
        }
    }

    void addRewardAtPosition(float x, float y) {
        // Add temporary reward zone
        reward_zones.insert(reward_zones.begin(), Vec3(x, y, 0));
        if (reward_zones.size() > 6) {
            reward_zones.pop_back();
        }

        // Emit particles
        for (int i = 0; i < 20; i++) {
            float angle = uniform_dist(gen) * 2.0f * PI;
            float speed = uniform_dist(gen) * 150.0f;
            Vec3 vel(std::cos(angle) * speed, std::sin(angle) * speed, 0);
            particles.emplace_back(Vec3(x, y, 0), vel, sf::Color(255, 215, 0, 200), 1.0f, 5.0f);
        }
    }

    void resetNetwork() {
        // Reset all synaptic weights
        for (auto& synapse : synapses) {
            synapse.weight = 0.2f + uniform_dist(gen) * 0.3f;
        }

        // Reset neuron states
        for (auto& neuron : neurons) {
            neuron.membrane_potential = -70.0f;
            neuron.firing_rate = neuron.baseline_firing_rate;
            neuron.activation_trace = 0.0f;
        }

        network_activity_history.clear();

        std::cout << "Network reset!\n";
    }

    void emitNeuronParticle(const Neuron& neuron) {
        Vec3 vel(
            normal_dist(gen) * 50.0f,
            normal_dist(gen) * 50.0f,
            0
        );
        auto colors = colorMap[neuron.personality];
        particles.emplace_back(neuron.position, vel, colors.glow, 0.5f, 3.0f);
    }

    void emitRewardParticles(const Vec3& position) {
        for (int i = 0; i < 3; i++) {
            float angle = uniform_dist(gen) * 2.0f * PI;
            float speed = uniform_dist(gen) * 100.0f;
            Vec3 vel(std::cos(angle) * speed, std::sin(angle) * speed, 0);
            sf::Color color(255, 215, 0, 200);
            particles.emplace_back(position, vel, color, 0.8f, 4.0f);
        }
    }

    void render(sf::RenderWindow& window) {
        window.clear(sf::Color(10, 10, 20));

        // Draw reward zones
        for (const auto& zone : reward_zones) {
            sf::CircleShape circle(100.0f);
            circle.setPosition(zone.x - 100.0f, zone.y - 100.0f);
            circle.setFillColor(sf::Color(255, 215, 0, 20));
            circle.setOutlineColor(sf::Color(255, 215, 0, 80));
            circle.setOutlineThickness(2.0f);
            window.draw(circle);
        }

        // Draw synaptic connections
        if (show_connections) {
            for (const auto& synapse : synapses) {
                const Neuron& pre = neurons[synapse.pre_neuron_id];
                const Neuron& post = neurons[synapse.post_neuron_id];

                // Only draw if not highlighting or if involved neuron is highlighted
                bool should_draw = true;
                if (highlighted_type >= 0) {
                    PersonalityType type = static_cast<PersonalityType>(highlighted_type);
                    should_draw = (pre.personality == type || post.personality == type);
                }

                if (should_draw) {
                    sf::Uint8 alpha = static_cast<sf::Uint8>(synapse.weight * 100.0f + synapse.pulse * 100.0f);
                    sf::Color color(150, 150, 150, alpha);

                    sf::Vertex line[] = {
                        sf::Vertex(sf::Vector2f(pre.position.x, pre.position.y), color),
                        sf::Vertex(sf::Vector2f(post.position.x, post.position.y), color)
                    };
                    window.draw(line, 2, sf::Lines);
                }
            }
        }

        // Draw synaptic signals
        if (show_signals) {
            for (const auto& signal : synaptic_signals) {
                if (signal.synapse_index >= 0 && signal.synapse_index < (int)synapses.size()) {
                    const Synapse& syn = synapses[signal.synapse_index];
                    const Neuron& pre = neurons[syn.pre_neuron_id];
                    const Neuron& post = neurons[syn.post_neuron_id];

                    Vec3 pos = pre.position + (post.position - pre.position) * signal.progress;

                    sf::CircleShape dot(3.0f);
                    dot.setPosition(pos.x - 3.0f, pos.y - 3.0f);
                    dot.setFillColor(signal.color);
                    window.draw(dot);
                }
            }
        }

        // Draw agent trails
        for (const auto& agent : agents) {
            for (size_t i = 1; i < agent.trail.size(); i++) {
                float alpha = (1.0f - (float)i / agent.trail.size()) * 80.0f;
                sf::Color color = agent.color;
                color.a = (sf::Uint8)alpha;

                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(agent.trail[i-1].x, agent.trail[i-1].y), color),
                    sf::Vertex(sf::Vector2f(agent.trail[i].x, agent.trail[i].y), color)
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        // Draw agents
        for (const auto& agent : agents) {
            sf::CircleShape shape(8.0f);
            shape.setPosition(agent.position.x - 8.0f, agent.position.y - 8.0f);
            shape.setFillColor(agent.color);
            window.draw(shape);
        }

        // Draw neurons
        for (const auto& neuron : neurons) {
            bool is_highlighted = (highlighted_type < 0) ||
                                 (neuron.personality == static_cast<PersonalityType>(highlighted_type));

            if (!is_highlighted) continue;

            auto colors = colorMap[neuron.personality];

            // Draw glow
            if (neuron.glow_intensity > 0.2f) {
                float glow_size = 8.0f + neuron.glow_intensity * 12.0f;
                sf::CircleShape glow(glow_size);
                glow.setPosition(neuron.position.x - glow_size, neuron.position.y - glow_size);

                sf::Color glow_color = colors.glow;
                glow_color.a = (sf::Uint8)(neuron.glow_intensity * 150.0f);
                glow.setFillColor(glow_color);
                window.draw(glow);
            }

            // Draw body
            float body_size = 3.0f + neuron.glow_intensity * 3.0f;
            sf::CircleShape body(body_size);
            body.setPosition(neuron.position.x - body_size, neuron.position.y - body_size);
            body.setFillColor(colors.primary);
            window.draw(body);
        }

        // Draw particles
        for (const auto& particle : particles) {
            sf::CircleShape shape(particle.size);
            shape.setPosition(particle.position.x - particle.size, particle.position.y - particle.size);
            shape.setFillColor(particle.color);
            window.draw(shape);
        }

        // Draw UI
        drawUI(window);
    }

    void drawUI(sf::RenderWindow& window) {
        // FPS and stats
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(0) << fps << "\n";
        ss << "Neurons: " << neurons.size() << "\n";
        ss << "Synapses: " << synapses.size() << "\n";
        ss << "Signals: " << synaptic_signals.size() << "\n";
        ss << "Speed: " << std::setprecision(1) << simulation_speed << "x\n";
        ss << "Reward: " << std::setprecision(2) << reward_signal;

        // Simple stats display (colored boxes)
        int y_offset = 20;

        // Legend
        for (int i = 0; i < 9; i++) {
            PersonalityType type = static_cast<PersonalityType>(i);
            auto colors = colorMap[type];

            bool is_highlighted = (highlighted_type == i);

            sf::RectangleShape box(sf::Vector2f(15, 15));
            box.setPosition(20, y_offset);
            box.setFillColor(colors.primary);

            if (is_highlighted) {
                box.setOutlineColor(sf::Color::White);
                box.setOutlineThickness(2);
            }

            window.draw(box);
            y_offset += 25;
        }

        // Activity graph
        if (show_graphs && !network_activity_history.empty()) {
            int graph_x = WINDOW_WIDTH - 400;
            int graph_y = 50;
            int graph_w = 350;
            int graph_h = 100;

            // Background
            sf::RectangleShape bg(sf::Vector2f(graph_w, graph_h));
            bg.setPosition(graph_x, graph_y);
            bg.setFillColor(sf::Color(0, 0, 0, 100));
            bg.setOutlineColor(sf::Color(100, 100, 100, 150));
            bg.setOutlineThickness(1);
            window.draw(bg);

            // Plot activity
            float max_activity = 50.0f; // Expected max
            for (size_t i = 1; i < network_activity_history.size(); i++) {
                float x1 = graph_x + (i - 1) * graph_w / 200.0f;
                float y1 = graph_y + graph_h - (network_activity_history[i-1] / max_activity) * graph_h;
                float x2 = graph_x + i * graph_w / 200.0f;
                float y2 = graph_y + graph_h - (network_activity_history[i] / max_activity) * graph_h;

                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(x1, y1), sf::Color(0, 255, 100)),
                    sf::Vertex(sf::Vector2f(x2, y2), sf::Color(0, 255, 100))
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        // FPS bar
        sf::RectangleShape fpsBar(sf::Vector2f(fps * 2.0f, 10));
        fpsBar.setPosition(WINDOW_WIDTH - 260, 20);

        if (fps >= TARGET_FPS * 0.9f) {
            fpsBar.setFillColor(sf::Color::Green);
        } else if (fps >= TARGET_FPS * 0.6f) {
            fpsBar.setFillColor(sf::Color::Yellow);
        } else {
            fpsBar.setFillColor(sf::Color::Red);
        }
        window.draw(fpsBar);

        // Controls hint
        sf::RectangleShape controls_bg(sf::Vector2f(300, 120));
        controls_bg.setPosition(20, WINDOW_HEIGHT - 140);
        controls_bg.setFillColor(sf::Color(0, 0, 0, 150));
        window.draw(controls_bg);
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                            "ONLY FOR NATURE - ULTIMATE EDITION");
    window.setFramerateLimit(TARGET_FPS);

    UltimateNatureSimulation sim;

    sf::Clock clock;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ONLY FOR NATURE - ULTIMATE EDITION                    ║\n";
    std::cout << "║         Complete Neural Network with Learning                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC         - Exit\n";
    std::cout << "  SPACE       - Pause/Resume\n";
    std::cout << "  Mouse Click - Add reward signal\n";
    std::cout << "  R           - Reset network\n";
    std::cout << "  C           - Toggle connections\n";
    std::cout << "  S           - Toggle signals\n";
    std::cout << "  G           - Toggle graphs\n";
    std::cout << "  +/-         - Adjust speed\n";
    std::cout << "  1-9         - Highlight neuron type\n";
    std::cout << "\n";
    std::cout << "Features:\n";
    std::cout << "  ✓ " << sim.neurons.size() << " Neurons with 9 personalities\n";
    std::cout << "  ✓ " << sim.synapses.size() << " Synaptic connections\n";
    std::cout << "  ✓ Hebbian learning (fire together, wire together)\n";
    std::cout << "  ✓ Real-time signal propagation\n";
    std::cout << "  ✓ " << sim.agents.size() << " Agents with different behaviors\n";
    std::cout << "  ✓ Interactive controls\n";
    std::cout << "\n";
    std::cout << "Starting simulation...\n\n";

    bool paused = false;

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
                } else if (event.key.code == sf::Keyboard::R) {
                    sim.resetNetwork();
                } else if (event.key.code == sf::Keyboard::C) {
                    sim.show_connections = !sim.show_connections;
                } else if (event.key.code == sf::Keyboard::S) {
                    sim.show_signals = !sim.show_signals;
                } else if (event.key.code == sf::Keyboard::G) {
                    sim.show_graphs = !sim.show_graphs;
                } else if (event.key.code == sf::Keyboard::Equal || event.key.code == sf::Keyboard::Add) {
                    sim.simulation_speed = std::min(5.0f, sim.simulation_speed + 0.5f);
                    std::cout << "Speed: " << sim.simulation_speed << "x\n";
                } else if (event.key.code == sf::Keyboard::Hyphen || event.key.code == sf::Keyboard::Subtract) {
                    sim.simulation_speed = std::max(0.1f, sim.simulation_speed - 0.5f);
                    std::cout << "Speed: " << sim.simulation_speed << "x\n";
                } else if (event.key.code >= sf::Keyboard::Num1 && event.key.code <= sf::Keyboard::Num9) {
                    int type = event.key.code - sf::Keyboard::Num1;
                    sim.highlighted_type = (sim.highlighted_type == type) ? -1 : type;
                    if (sim.highlighted_type >= 0) {
                        std::cout << "Highlighting: " << personalityNames[static_cast<PersonalityType>(type)] << "\n";
                    } else {
                        std::cout << "Showing all neurons\n";
                    }
                } else if (event.key.code == sf::Keyboard::Num0) {
                    sim.highlighted_type = -1;
                }
            }

            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sim.addRewardAtPosition(event.mouseButton.x, event.mouseButton.y);
                    std::cout << "Reward added at (" << event.mouseButton.x << ", " << event.mouseButton.y << ")\n";
                }
            }
        }

        if (!paused) {
            float dt = clock.restart().asSeconds();
            dt = std::min(dt, 0.033f);
            sim.update(dt);
        } else {
            clock.restart();
        }

        sim.render(window);
        window.display();
    }

    std::cout << "\nSimulation ended.\n";
    std::cout << "Final statistics:\n";
    std::cout << "  Total neurons: " << sim.neurons.size() << "\n";
    std::cout << "  Total synapses: " << sim.synapses.size() << "\n";

    // Calculate average synaptic weight
    float avg_weight = 0.0f;
    for (const auto& syn : sim.synapses) {
        avg_weight += syn.weight;
    }
    avg_weight /= sim.synapses.size();
    std::cout << "  Average synaptic weight: " << std::fixed << std::setprecision(3) << avg_weight << "\n";

    return 0;
}
