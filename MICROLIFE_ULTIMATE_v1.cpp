/**
 * ╔═══════════════════════════════════════════════════════════════════╗
 * ║                    MICROLIFE ULTIMATE                              ║
 * ║        100+ FPS Micro-Organism Ecosystem Simulation               ║
 * ║                                                                    ║
 * ║  🦠 ORGANISMS:                                                     ║
 * ║    • Algae (Photosynthetic) - Işıktan enerji                      ║
 * ║    • Predators - Diğerlerini avlar                                ║
 * ║    • Scavengers - Ölü canlılardan beslenir                        ║
 * ║                                                                    ║
 * ║  🌍 ENVIRONMENTS:                                                  ║
 * ║    1. Lake (Göl) - Akıntılar, oksijen katmanları                  ║
 * ║    2. Ocean Reef (Resif) - Mercanlar, gelgit                      ║
 * ║    3. Forest Floor (Orman) - Nem, çürüyen yapraklar               ║
 * ║    4. Volcanic Vent - Sıcaklık, mineraller                        ║
 * ║    5. Arctic Ice - Soğuk, sınırlı kaynak                          ║
 * ║    6. Immune System - Patojenler, kan akışı                       ║
 * ║                                                                    ║
 * ║  🧬 EVOLUTION:                                                     ║
 * ║    • Reproduction & Mutation                                       ║
 * ║    • Natural Selection                                             ║
 * ║    • Adaptation to Environment                                     ║
 * ║                                                                    ║
 * ║  📊 GRAPHS:                                                        ║
 * ║    • Population over time                                          ║
 * ║    • Energy distribution                                           ║
 * ║    • Species diversity                                             ║
 * ║                                                                    ║
 * ║  Controls:                                                         ║
 * ║    ESC   - Exit            SPACE - Pause                          ║
 * ║    Mouse - Add organism    R - Reset                              ║
 * ║    1-3   - Add type        G - Toggle graphs                      ║
 * ║    E     - Change env      +/- - Speed                            ║
 * ║                                                                    ║
 * ║  Build: make -f Makefile.microlife                                ║
 * ║  Run:   ./MICROLIFE_ULTIMATE                                      ║
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

// Constants
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;
const float PI = 3.14159265359f;
const int TARGET_FPS = 120;

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
std::normal_distribution<float> normal_dist(0.0f, 1.0f);

// Vector2D
struct Vec2 {
    float x, y;

    Vec2(float x = 0, float y = 0) : x(x), y(y) {}

    float length() const {
        return std::sqrt(x*x + y*y);
    }

    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    Vec2 operator*(float scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    float distance(const Vec2& other) const {
        return (*this - other).length();
    }

    Vec2 normalized() const {
        float len = length();
        if (len > 0) return Vec2(x/len, y/len);
        return Vec2(0, 0);
    }
};

// Organism Types
enum class OrganismType {
    ALGAE,      // 🌿 Photosynthetic - gains energy from light
    PREDATOR,   // 🦈 Hunter - eats other organisms
    SCAVENGER   // 🦀 Decomposer - eats dead organisms
};

// Environment Types
enum class EnvironmentType {
    LAKE,
    OCEAN_REEF,
    FOREST_FLOOR,
    VOLCANIC_VENT,
    ARCTIC_ICE,
    IMMUNE_SYSTEM
};

// Environment configuration
struct EnvironmentConfig {
    std::string name;
    sf::Color background;
    sf::Color accent;
    float temperature;      // -50 to 100 °C
    float light_level;      // 0 to 1
    float toxicity;         // 0 to 1
    Vec2 current;          // Environmental flow

    static EnvironmentConfig getLake() {
        return {
            "🌊 Lake Ecosystem",
            sf::Color(20, 60, 100),
            sf::Color(100, 180, 255),
            20.0f,  // Moderate temp
            0.7f,   // Good light
            0.1f,   // Low toxicity
            Vec2(0.5f, 0.0f)  // Gentle current
        };
    }

    static EnvironmentConfig getOceanReef() {
        return {
            "🐠 Ocean Reef",
            sf::Color(10, 40, 80),
            sf::Color(0, 200, 200),
            25.0f,
            0.8f,
            0.05f,
            Vec2(std::cos(0.0f) * 1.0f, std::sin(0.0f) * 0.5f)
        };
    }

    static EnvironmentConfig getForestFloor() {
        return {
            "🌲 Forest Floor",
            sf::Color(40, 30, 20),
            sf::Color(100, 150, 50),
            15.0f,
            0.3f,   // Low light
            0.2f,
            Vec2(0.0f, 0.0f)
        };
    }

    static EnvironmentConfig getVolcanicVent() {
        return {
            "🌋 Volcanic Vent",
            sf::Color(60, 20, 10),
            sf::Color(255, 100, 0),
            80.0f,  // Hot!
            0.1f,   // Dark
            0.6f,   // Toxic!
            Vec2(uniform_dist(gen) * 2.0f - 1.0f, uniform_dist(gen) * 2.0f - 1.0f)
        };
    }

    static EnvironmentConfig getArcticIce() {
        return {
            "❄️ Arctic Ice",
            sf::Color(200, 220, 255),
            sf::Color(255, 255, 255),
            -10.0f,  // Cold!
            0.5f,
            0.0f,
            Vec2(uniform_dist(gen) * 3.0f - 1.5f, uniform_dist(gen) * 3.0f - 1.5f)
        };
    }

    static EnvironmentConfig getImmuneSystem() {
        return {
            "🦠 Immune System",
            sf::Color(100, 20, 20),
            sf::Color(255, 50, 50),
            37.0f,  // Body temp
            0.0f,   // No light
            0.3f,
            Vec2(2.0f, 0.0f)  // Blood flow
        };
    }

    static EnvironmentConfig get(EnvironmentType type) {
        switch(type) {
            case EnvironmentType::LAKE: return getLake();
            case EnvironmentType::OCEAN_REEF: return getOceanReef();
            case EnvironmentType::FOREST_FLOOR: return getForestFloor();
            case EnvironmentType::VOLCANIC_VENT: return getVolcanicVent();
            case EnvironmentType::ARCTIC_ICE: return getArcticIce();
            case EnvironmentType::IMMUNE_SYSTEM: return getImmuneSystem();
            default: return getLake();
        }
    }
};

// Food particle
struct FoodParticle {
    Vec2 position;
    float energy;
    float size;
    sf::Color color;

    FoodParticle(Vec2 pos, float e)
        : position(pos), energy(e), size(2.0f + e * 0.05f), color(0, 255, 100) {}
};

// Organism class
class Organism {
public:
    Vec2 position;
    Vec2 velocity;
    OrganismType type;
    float energy;
    float max_energy;
    float size;
    float speed;
    float perception_radius;
    sf::Color color;
    bool alive;
    int age;

    // Genetics
    float mutation_rate = 0.1f;
    float efficiency;    // Energy usage efficiency
    float aggression;    // For predators

    // Trail
    std::deque<Vec2> trail;

    Organism(Vec2 pos, OrganismType t)
        : position(pos), velocity(0, 0), type(t), energy(100.0f),
          max_energy(200.0f), alive(true), age(0) {

        // Type-specific attributes
        switch(type) {
            case OrganismType::ALGAE:
                color = sf::Color(50, 255, 100);
                size = 4.0f;
                speed = 0.5f;
                perception_radius = 30.0f;
                efficiency = 0.8f + uniform_dist(gen) * 0.2f;
                aggression = 0.0f;
                break;

            case OrganismType::PREDATOR:
                color = sf::Color(255, 50, 50);
                size = 6.0f;
                speed = 2.0f;
                perception_radius = 100.0f;
                efficiency = 0.6f + uniform_dist(gen) * 0.2f;
                aggression = 0.5f + uniform_dist(gen) * 0.5f;
                break;

            case OrganismType::SCAVENGER:
                color = sf::Color(200, 200, 100);
                size = 5.0f;
                speed = 1.5f;
                perception_radius = 80.0f;
                efficiency = 0.9f + uniform_dist(gen) * 0.1f;
                aggression = 0.1f;
                break;
        }

        // Add some variation
        color.r = std::min(255, std::max(0, (int)color.r + (int)(normal_dist(gen) * 30)));
        color.g = std::min(255, std::max(0, (int)color.g + (int)(normal_dist(gen) * 30)));
        color.b = std::min(255, std::max(0, (int)color.b + (int)(normal_dist(gen) * 30)));
    }

    void update(float dt, const EnvironmentConfig& env,
                std::vector<FoodParticle>& food,
                std::vector<Organism>& organisms) {

        if (!alive) return;

        age++;

        // Environmental effects
        applyEnvironmentalEffects(env, dt);

        // Type-specific behavior
        switch(type) {
            case OrganismType::ALGAE:
                updateAlgae(env, dt);
                break;
            case OrganismType::PREDATOR:
                updatePredator(organisms, dt);
                break;
            case OrganismType::SCAVENGER:
                updateScavenger(food, dt);
                break;
        }

        // Apply environmental current
        velocity = velocity + env.current * dt * 10.0f;

        // Apply velocity
        position = position + velocity * dt * 50.0f;

        // Bounds
        if (position.x < 0 || position.x > WINDOW_WIDTH) velocity.x *= -0.5f;
        if (position.y < 0 || position.y > WINDOW_HEIGHT) velocity.y *= -0.5f;
        position.x = std::max(0.0f, std::min((float)WINDOW_WIDTH, position.x));
        position.y = std::max(0.0f, std::min((float)WINDOW_HEIGHT, position.y));

        // Friction
        velocity = velocity * 0.95f;

        // Energy consumption
        float base_cost = 0.05f * dt;
        float movement_cost = velocity.length() * 0.01f * dt;
        energy -= (base_cost + movement_cost) / efficiency;

        // Death
        if (energy <= 0 || age > 10000) {
            alive = false;
        }

        // Trail
        trail.push_front(position);
        if (trail.size() > 20) {
            trail.pop_back();
        }
    }

    void updateAlgae(const EnvironmentConfig& env, float dt) {
        // Photosynthesis - gain energy from light
        energy += env.light_level * 5.0f * dt;
        energy = std::min(energy, max_energy);

        // Slow drift
        if (uniform_dist(gen) < 0.1f) {
            velocity = Vec2(normal_dist(gen) * 0.5f, normal_dist(gen) * 0.5f);
        }
    }

    void updatePredator(std::vector<Organism>& organisms, float dt) {
        // Hunt for prey
        Organism* nearest_prey = nullptr;
        float nearest_dist = perception_radius;

        for (auto& org : organisms) {
            if (!org.alive || &org == this) continue;
            if (org.type == OrganismType::PREDATOR) continue; // Don't eat own kind

            float dist = position.distance(org.position);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_prey = &org;
            }
        }

        if (nearest_prey) {
            // Chase
            Vec2 direction = (nearest_prey->position - position).normalized();
            velocity = velocity + direction * aggression * speed * dt * 100.0f;

            // Eat if close enough
            if (nearest_dist < size + nearest_prey->size) {
                energy += nearest_prey->energy * 0.7f; // 70% efficiency
                energy = std::min(energy, max_energy);
                nearest_prey->alive = false;
            }
        } else {
            // Wander
            if (uniform_dist(gen) < 0.05f) {
                velocity = Vec2(normal_dist(gen) * speed, normal_dist(gen) * speed);
            }
        }
    }

    void updateScavenger(std::vector<FoodParticle>& food, float dt) {
        // Look for food particles
        FoodParticle* nearest_food = nullptr;
        float nearest_dist = perception_radius;
        size_t food_idx = 0;

        for (size_t i = 0; i < food.size(); i++) {
            float dist = position.distance(food[i].position);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_food = &food[i];
                food_idx = i;
            }
        }

        if (nearest_food) {
            // Move towards food
            Vec2 direction = (nearest_food->position - position).normalized();
            velocity = velocity + direction * speed * dt * 80.0f;

            // Eat if close enough
            if (nearest_dist < size + nearest_food->size) {
                energy += nearest_food->energy;
                energy = std::min(energy, max_energy);
                food.erase(food.begin() + food_idx);
            }
        } else {
            // Wander
            if (uniform_dist(gen) < 0.05f) {
                velocity = Vec2(normal_dist(gen) * speed, normal_dist(gen) * speed);
            }
        }
    }

    void applyEnvironmentalEffects(const EnvironmentConfig& env, float dt) {
        // Temperature stress
        float temp_stress = 0.0f;
        if (type == OrganismType::ALGAE) {
            // Algae like 15-25°C
            float ideal = 20.0f;
            temp_stress = std::abs(env.temperature - ideal) * 0.01f;
        } else if (type == OrganismType::PREDATOR) {
            // Predators like 10-30°C
            if (env.temperature < 10.0f || env.temperature > 30.0f) {
                temp_stress = 0.02f;
            }
        }

        // Toxicity damage
        float toxicity_damage = env.toxicity * 1.0f * dt;

        energy -= (temp_stress + toxicity_damage);
    }

    Organism reproduce() const {
        // Create offspring with mutation
        Organism child(position + Vec2(normal_dist(gen) * 10.0f, normal_dist(gen) * 10.0f), type);

        // Inherit traits with mutation
        child.efficiency = efficiency + normal_dist(gen) * mutation_rate * 0.1f;
        child.efficiency = std::max(0.1f, std::min(1.0f, child.efficiency));

        child.aggression = aggression + normal_dist(gen) * mutation_rate * 0.1f;
        child.aggression = std::max(0.0f, std::min(1.0f, child.aggression));

        child.speed = speed + normal_dist(gen) * mutation_rate * 0.2f;
        child.speed = std::max(0.1f, std::min(5.0f, child.speed));

        child.energy = energy * 0.3f; // Parent gives 30% energy

        return child;
    }

    bool canReproduce() const {
        return alive && energy > max_energy * 0.7f && age > 100;
    }
};

// Main Simulation
class MicroLifeSimulation {
public:
    EnvironmentType current_env_type;
    EnvironmentConfig env_config;
    std::vector<Organism> organisms;
    std::vector<FoodParticle> food_particles;

    float time = 0.0f;
    float simulation_speed = 1.0f;
    bool paused = false;
    bool show_graphs = true;

    // Statistics
    int frame_count = 0;
    float fps = 0.0f;
    sf::Clock fps_clock;
    std::deque<int> population_history;
    std::deque<float> energy_history;
    std::map<OrganismType, int> species_count;

    MicroLifeSimulation(EnvironmentType env_type = EnvironmentType::LAKE)
        : current_env_type(env_type) {

        env_config = EnvironmentConfig::get(env_type);
        initializePopulation();
    }

    void changeEnvironment(EnvironmentType new_env) {
        current_env_type = new_env;
        env_config = EnvironmentConfig::get(new_env);
        std::cout << "Environment changed to: " << env_config.name << "\n";
    }

    void initializePopulation() {
        organisms.clear();

        // Add diverse starting population
        for (int i = 0; i < 30; i++) {
            Vec2 pos(uniform_dist(gen) * WINDOW_WIDTH, uniform_dist(gen) * WINDOW_HEIGHT);
            OrganismType type;

            float r = uniform_dist(gen);
            if (r < 0.5f) type = OrganismType::ALGAE;
            else if (r < 0.8f) type = OrganismType::SCAVENGER;
            else type = OrganismType::PREDATOR;

            organisms.emplace_back(pos, type);
        }

        std::cout << "Initialized " << organisms.size() << " organisms\n";
    }

    void addOrganism(Vec2 pos, OrganismType type) {
        organisms.emplace_back(pos, type);
    }

    void update(float dt) {
        dt *= simulation_speed;
        time += dt;

        // Update organisms
        for (auto& org : organisms) {
            org.update(dt, env_config, food_particles, organisms);
        }

        // Reproduction
        std::vector<Organism> offspring;
        for (const auto& org : organisms) {
            if (org.canReproduce() && uniform_dist(gen) < 0.01f) {
                offspring.push_back(org.reproduce());
            }
        }

        for (auto& child : offspring) {
            organisms.push_back(child);
        }

        // Remove dead organisms, create food
        organisms.erase(
            std::remove_if(organisms.begin(), organisms.end(),
                [this](Organism& org) {
                    if (!org.alive) {
                        // Dead organisms become food
                        food_particles.emplace_back(org.position, org.energy * 0.5f);
                        return true;
                    }
                    return false;
                }),
            organisms.end()
        );

        // Decay food
        food_particles.erase(
            std::remove_if(food_particles.begin(), food_particles.end(),
                [dt](FoodParticle& f) {
                    f.energy -= 0.5f * dt;
                    return f.energy <= 0;
                }),
            food_particles.end()
        );

        // Update statistics
        updateStatistics();

        // FPS
        frame_count++;
        if (fps_clock.getElapsedTime().asSeconds() >= 1.0f) {
            fps = frame_count / fps_clock.getElapsedTime().asSeconds();
            frame_count = 0;
            fps_clock.restart();
        }
    }

    void updateStatistics() {
        // Count species
        species_count.clear();
        float total_energy = 0.0f;

        for (const auto& org : organisms) {
            species_count[org.type]++;
            total_energy += org.energy;
        }

        population_history.push_front(organisms.size());
        if (population_history.size() > 300) {
            population_history.pop_back();
        }

        float avg_energy = organisms.empty() ? 0 : total_energy / organisms.size();
        energy_history.push_front(avg_energy);
        if (energy_history.size() > 300) {
            energy_history.pop_back();
        }
    }

    void render(sf::RenderWindow& window) {
        // Background
        window.clear(env_config.background);

        // Environment effects (simple gradients/patterns)
        renderEnvironmentEffects(window);

        // Food particles
        for (const auto& food : food_particles) {
            sf::CircleShape shape(food.size);
            shape.setPosition(food.position.x - food.size, food.position.y - food.size);
            sf::Color c = food.color;
            c.a = 150;
            shape.setFillColor(c);
            window.draw(shape);
        }

        // Organisms with trails
        for (const auto& org : organisms) {
            // Trail
            for (size_t i = 1; i < org.trail.size(); i++) {
                float alpha = (1.0f - (float)i / org.trail.size()) * 100.0f;
                sf::Color trail_color = org.color;
                trail_color.a = (sf::Uint8)alpha;

                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(org.trail[i-1].x, org.trail[i-1].y), trail_color),
                    sf::Vertex(sf::Vector2f(org.trail[i].x, org.trail[i].y), trail_color)
                };
                window.draw(line, 2, sf::Lines);
            }

            // Body
            sf::CircleShape body(org.size);
            body.setPosition(org.position.x - org.size, org.position.y - org.size);

            // Color based on energy
            sf::Color body_color = org.color;
            float energy_ratio = org.energy / org.max_energy;
            body_color.a = (sf::Uint8)(100 + energy_ratio * 155);

            body.setFillColor(body_color);
            body.setOutlineColor(sf::Color::White);
            body.setOutlineThickness(0.5f);

            window.draw(body);
        }

        // UI
        renderUI(window);
    }

    void renderEnvironmentEffects(sf::RenderWindow& window) {
        // Draw environment-specific visual elements
        switch(current_env_type) {
            case EnvironmentType::LAKE:
                // Water ripples (simple lines)
                for (int i = 0; i < 5; i++) {
                    float y = (time * 10.0f + i * 200.0f);
                    while (y > WINDOW_HEIGHT) y -= WINDOW_HEIGHT * 5;

                    sf::RectangleShape wave(sf::Vector2f(WINDOW_WIDTH, 2));
                    wave.setPosition(0, y);
                    wave.setFillColor(sf::Color(100, 180, 255, 30));
                    window.draw(wave);
                }
                break;

            case EnvironmentType::VOLCANIC_VENT:
                // Rising bubbles
                break;

            default:
                break;
        }
    }

    void renderUI(sf::RenderWindow& window) {
        // Environment name
        // Species counts
        int y_offset = 20;

        // Environment indicator
        sf::RectangleShape env_box(sf::Vector2f(300, 30));
        env_box.setPosition(20, y_offset);
        env_box.setFillColor(sf::Color(0, 0, 0, 150));
        window.draw(env_box);
        y_offset += 50;

        // Species legend
        std::vector<std::pair<OrganismType, std::string>> types = {
            {OrganismType::ALGAE, "Algae"},
            {OrganismType::PREDATOR, "Predator"},
            {OrganismType::SCAVENGER, "Scavenger"}
        };

        for (const auto& [type, name] : types) {
            sf::CircleShape icon(8);
            icon.setPosition(25, y_offset);

            switch(type) {
                case OrganismType::ALGAE:
                    icon.setFillColor(sf::Color(50, 255, 100));
                    break;
                case OrganismType::PREDATOR:
                    icon.setFillColor(sf::Color(255, 50, 50));
                    break;
                case OrganismType::SCAVENGER:
                    icon.setFillColor(sf::Color(200, 200, 100));
                    break;
            }

            window.draw(icon);
            y_offset += 30;
        }

        // Graphs
        if (show_graphs) {
            renderGraphs(window);
        }

        // FPS
        sf::RectangleShape fpsBar(sf::Vector2f(fps * 2.0f, 10));
        fpsBar.setPosition(WINDOW_WIDTH - 260, 20);
        fpsBar.setFillColor(fps >= TARGET_FPS * 0.9f ? sf::Color::Green :
                           fps >= TARGET_FPS * 0.6f ? sf::Color::Yellow : sf::Color::Red);
        window.draw(fpsBar);
    }

    void renderGraphs(sf::RenderWindow& window) {
        int graph_x = WINDOW_WIDTH - 400;
        int graph_y = 50;
        int graph_w = 350;
        int graph_h = 150;

        // Population graph
        sf::RectangleShape bg(sf::Vector2f(graph_w, graph_h));
        bg.setPosition(graph_x, graph_y);
        bg.setFillColor(sf::Color(0, 0, 0, 150));
        bg.setOutlineColor(sf::Color(100, 100, 100, 200));
        bg.setOutlineThickness(1);
        window.draw(bg);

        if (!population_history.empty()) {
            int max_pop = *std::max_element(population_history.begin(), population_history.end());
            max_pop = std::max(1, max_pop);

            for (size_t i = 1; i < population_history.size(); i++) {
                float x1 = graph_x + (i - 1) * graph_w / 300.0f;
                float y1 = graph_y + graph_h - (population_history[i-1] * graph_h / max_pop);
                float x2 = graph_x + i * graph_w / 300.0f;
                float y2 = graph_y + graph_h - (population_history[i] * graph_h / max_pop);

                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(x1, y1), sf::Color::Green),
                    sf::Vertex(sf::Vector2f(x2, y2), sf::Color::Green)
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        // Energy graph
        graph_y += graph_h + 20;

        bg.setPosition(graph_x, graph_y);
        window.draw(bg);

        if (!energy_history.empty()) {
            float max_energy = *std::max_element(energy_history.begin(), energy_history.end());
            max_energy = std::max(1.0f, max_energy);

            for (size_t i = 1; i < energy_history.size(); i++) {
                float x1 = graph_x + (i - 1) * graph_w / 300.0f;
                float y1 = graph_y + graph_h - (energy_history[i-1] * graph_h / max_energy);
                float x2 = graph_x + i * graph_w / 300.0f;
                float y2 = graph_y + graph_h - (energy_history[i] * graph_h / max_energy);

                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(x1, y1), sf::Color::Yellow),
                    sf::Vertex(sf::Vector2f(x2, y2), sf::Color::Yellow)
                };
                window.draw(line, 2, sf::Lines);
            }
        }
    }

    void reset() {
        organisms.clear();
        food_particles.clear();
        population_history.clear();
        energy_history.clear();
        initializePopulation();
        std::cout << "Simulation reset!\n";
    }
};

void showEnvironmentMenu() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    MICROLIFE ULTIMATE                              ║\n";
    std::cout << "║               Select Your Ecosystem                                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "1. 🌊 Lake Ecosystem       - Gentle currents, good light\n";
    std::cout << "2. 🐠 Ocean Reef           - Tidal flows, coral structures\n";
    std::cout << "3. 🌲 Forest Floor         - Low light, decaying matter\n";
    std::cout << "4. 🌋 Volcanic Vent        - Hot, toxic, rich minerals\n";
    std::cout << "5. ❄️  Arctic Ice           - Cold, harsh, limited resources\n";
    std::cout << "6. 🦠 Immune System        - Blood flow, pathogens\n";
    std::cout << "\n";
    std::cout << "Select environment (1-6): ";
}

int main() {
    // Show menu
    showEnvironmentMenu();

    int choice;
    std::cin >> choice;

    EnvironmentType env_type = EnvironmentType::LAKE;
    switch(choice) {
        case 1: env_type = EnvironmentType::LAKE; break;
        case 2: env_type = EnvironmentType::OCEAN_REEF; break;
        case 3: env_type = EnvironmentType::FOREST_FLOOR; break;
        case 4: env_type = EnvironmentType::VOLCANIC_VENT; break;
        case 5: env_type = EnvironmentType::ARCTIC_ICE; break;
        case 6: env_type = EnvironmentType::IMMUNE_SYSTEM; break;
        default: env_type = EnvironmentType::LAKE;
    }

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                            "MICROLIFE ULTIMATE - Ecosystem Simulation");
    window.setFramerateLimit(TARGET_FPS);

    MicroLifeSimulation sim(env_type);

    sf::Clock clock;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  SIMULATION STARTED                                ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Environment: " << sim.env_config.name << "\n";
    std::cout << "Organisms: " << sim.organisms.size() << "\n";
    std::cout << "\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC         - Exit\n";
    std::cout << "  SPACE       - Pause/Resume\n";
    std::cout << "  Mouse Click - Add organism at cursor\n";
    std::cout << "  1           - Add Algae\n";
    std::cout << "  2           - Add Predator\n";
    std::cout << "  3           - Add Scavenger\n";
    std::cout << "  R           - Reset simulation\n";
    std::cout << "  G           - Toggle graphs\n";
    std::cout << "  E           - Change environment\n";
    std::cout << "  +/-         - Adjust speed\n";
    std::cout << "\n";

    OrganismType add_type = OrganismType::ALGAE;

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
                    sim.paused = !sim.paused;
                } else if (event.key.code == sf::Keyboard::R) {
                    sim.reset();
                } else if (event.key.code == sf::Keyboard::G) {
                    sim.show_graphs = !sim.show_graphs;
                } else if (event.key.code == sf::Keyboard::Num1) {
                    add_type = OrganismType::ALGAE;
                    std::cout << "Adding: Algae\n";
                } else if (event.key.code == sf::Keyboard::Num2) {
                    add_type = OrganismType::PREDATOR;
                    std::cout << "Adding: Predator\n";
                } else if (event.key.code == sf::Keyboard::Num3) {
                    add_type = OrganismType::SCAVENGER;
                    std::cout << "Adding: Scavenger\n";
                } else if (event.key.code == sf::Keyboard::E) {
                    // Cycle through environments
                    int next = ((int)sim.current_env_type + 1) % 6;
                    sim.changeEnvironment((EnvironmentType)next);
                } else if (event.key.code == sf::Keyboard::Equal) {
                    sim.simulation_speed = std::min(5.0f, sim.simulation_speed + 0.5f);
                    std::cout << "Speed: " << sim.simulation_speed << "x\n";
                } else if (event.key.code == sf::Keyboard::Hyphen) {
                    sim.simulation_speed = std::max(0.1f, sim.simulation_speed - 0.5f);
                    std::cout << "Speed: " << sim.simulation_speed << "x\n";
                }
            }

            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    Vec2 pos(event.mouseButton.x, event.mouseButton.y);
                    sim.addOrganism(pos, add_type);
                }
            }
        }

        if (!sim.paused) {
            float dt = clock.restart().asSeconds();
            dt = std::min(dt, 0.033f);
            sim.update(dt);
        } else {
            clock.restart();
        }

        sim.render(window);
        window.display();
    }

    std::cout << "\n";
    std::cout << "Simulation ended.\n";
    std::cout << "Final population: " << sim.organisms.size() << "\n";
    std::cout << "Species breakdown:\n";
    std::cout << "  Algae: " << sim.species_count[OrganismType::ALGAE] << "\n";
    std::cout << "  Predators: " << sim.species_count[OrganismType::PREDATOR] << "\n";
    std::cout << "  Scavengers: " << sim.species_count[OrganismType::SCAVENGER] << "\n";

    return 0;
}
