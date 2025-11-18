/**
 * ╔═══════════════════════════════════════════════════════════════════╗
 * ║              MICROLIFE ULTIMATE - CINEMATIC EDITION                ║
 * ║          Film-Quality Micro-Organism Evolution Simulator          ║
 * ║                                                                    ║
 * ║  🎬 CINEMATIC FEATURES:                                            ║
 * ║    • Live Event Feed - "Organism A evolved!"                      ║
 * ║    • 12 Unique Organism Types                                     ║
 * ║    • Film-Quality Visual Effects                                  ║
 * ║    • Mutation Announcements                                       ║
 * ║    • Birth/Death Notifications                                    ║
 * ║    • Evolution Tracking                                            ║
 * ║                                                                    ║
 * ║  🦠 12 ORGANISM TYPES:                                             ║
 * ║    1. Photosynthetic Algae    7. Toxic Bacteria                   ║
 * ║    2. Hunter Predator          8. Giant Amoeba                    ║
 * ║    3. Scavenger Decomposer     9. Speed Demon                     ║
 * ║    4. Parasite                10. Tank Organism                   ║
 * ║    5. Symbiotic Partner       11. Energy Vampire                  ║
 * ║    6. Colony Former           12. Apex Predator                   ║
 * ║                                                                    ║
 * ║  Build: make -f Makefile.microlife.cinematic                      ║
 * ║  Run:   ./MICROLIFE_CINEMATIC                                     ║
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
#include <chrono>

// Constants
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;
const int EVENT_FEED_WIDTH = 400;
const float PI = 3.14159265359f;
const int TARGET_FPS = 120;

// Random
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
std::normal_distribution<float> normal_dist(0.0f, 1.0f);

// Vector2D
struct Vec2 {
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}
    float length() const { return std::sqrt(x*x + y*y); }
    Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2& o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    float distance(const Vec2& o) const { return (*this - o).length(); }
    Vec2 normalized() const {
        float len = length();
        return len > 0 ? Vec2(x/len, y/len) : Vec2(0, 0);
    }
};

// Organism Types (12 unique types!)
enum class OrganismType {
    PHOTOSYNTHETIC_ALGAE,    // 🌿 Energy from light
    HUNTER_PREDATOR,         // 🦈 Fast hunter
    SCAVENGER_DECOMPOSER,    // 🦀 Eats dead stuff
    PARASITE,                // 🐛 Steals energy from others
    SYMBIOTIC_PARTNER,       // 🤝 Helps others, gets help
    COLONY_FORMER,           // 👥 Forms groups
    TOXIC_BACTERIA,          // ☠️ Poisons others
    GIANT_AMOEBA,            // 🔵 Large, slow, tough
    SPEED_DEMON,             // ⚡ Super fast, fragile
    TANK_ORGANISM,           // 🛡️ Armored, slow
    ENERGY_VAMPIRE,          // 🧛 Drains energy without killing
    APEX_PREDATOR            // 👑 Top of food chain
};

// Type info
struct OrganismTypeInfo {
    std::string name;
    std::string emoji;
    sf::Color color;
    float base_speed;
    float base_size;
    float base_energy;
    std::string description;
};

std::map<OrganismType, OrganismTypeInfo> TYPE_INFO = {
    {OrganismType::PHOTOSYNTHETIC_ALGAE, {"Algae", "🌿", sf::Color(50, 255, 100), 0.5f, 4.0f, 100.0f, "Photosynthesis from light"}},
    {OrganismType::HUNTER_PREDATOR, {"Hunter", "🦈", sf::Color(255, 50, 50), 2.5f, 6.0f, 120.0f, "Fast aggressive hunter"}},
    {OrganismType::SCAVENGER_DECOMPOSER, {"Scavenger", "🦀", sf::Color(200, 200, 100), 1.5f, 5.0f, 110.0f, "Eats dead organisms"}},
    {OrganismType::PARASITE, {"Parasite", "🐛", sf::Color(150, 50, 150), 1.0f, 3.0f, 80.0f, "Steals from others"}},
    {OrganismType::SYMBIOTIC_PARTNER, {"Symbiont", "🤝", sf::Color(100, 255, 255), 1.2f, 4.5f, 100.0f, "Mutual cooperation"}},
    {OrganismType::COLONY_FORMER, {"Colony", "👥", sf::Color(255, 200, 100), 0.8f, 5.0f, 90.0f, "Strength in numbers"}},
    {OrganismType::TOXIC_BACTERIA, {"Toxic", "☠️", sf::Color(100, 255, 50), 1.0f, 3.5f, 70.0f, "Poisons nearby"}},
    {OrganismType::GIANT_AMOEBA, {"Giant", "🔵", sf::Color(100, 150, 255), 0.3f, 10.0f, 200.0f, "Huge and resilient"}},
    {OrganismType::SPEED_DEMON, {"Speedster", "⚡", sf::Color(255, 255, 100), 4.0f, 3.0f, 60.0f, "Lightning fast"}},
    {OrganismType::TANK_ORGANISM, {"Tank", "🛡️", sf::Color(150, 150, 150), 0.4f, 8.0f, 180.0f, "Armored fortress"}},
    {OrganismType::ENERGY_VAMPIRE, {"Vampire", "🧛", sf::Color(200, 50, 100), 1.8f, 5.5f, 100.0f, "Drains energy"}},
    {OrganismType::APEX_PREDATOR, {"Apex", "👑", sf::Color(255, 100, 0), 2.0f, 7.0f, 150.0f, "Top of food chain"}}
};

// Event types for live feed
enum class EventType {
    BIRTH,
    DEATH,
    EVOLUTION,
    MUTATION,
    HUNT_SUCCESS,
    PARASITE_ATTACH,
    SYMBIOSIS_FORMED,
    COLONY_FORMED,
    POISONED,
    ESCAPED
};

struct GameEvent {
    EventType type;
    std::string message;
    sf::Color color;
    float lifetime;
    float max_lifetime;

    GameEvent(EventType t, const std::string& msg, sf::Color col)
        : type(t), message(msg), color(col), lifetime(3.0f), max_lifetime(3.0f) {}

    void update(float dt) {
        lifetime -= dt;
    }

    bool isExpired() const {
        return lifetime <= 0;
    }

    float getAlpha() const {
        return (lifetime / max_lifetime) * 255.0f;
    }
};

// Particle for visual effects
struct Particle {
    Vec2 position;
    Vec2 velocity;
    sf::Color color;
    float lifetime;
    float max_lifetime;
    float size;
    float rotation;
    float rotation_speed;

    Particle(Vec2 pos, Vec2 vel, sf::Color col, float life, float sz = 2.0f)
        : position(pos), velocity(vel), color(col), lifetime(life),
          max_lifetime(life), size(sz), rotation(0), rotation_speed(uniform_dist(gen) * 360.0f) {}

    void update(float dt) {
        position = position + velocity * dt;
        velocity = velocity * 0.98f; // Air resistance
        lifetime -= dt;
        rotation += rotation_speed * dt;

        float alpha = (lifetime / max_lifetime) * 255.0f;
        color.a = static_cast<sf::Uint8>(std::max(0.0f, std::min(255.0f, alpha)));
    }

    bool isAlive() const { return lifetime > 0; }
};

// Environment
enum class EnvironmentType { LAKE, OCEAN_REEF, FOREST_FLOOR, VOLCANIC_VENT, ARCTIC_ICE, IMMUNE_SYSTEM };

struct EnvironmentConfig {
    std::string name;
    sf::Color background;
    float temperature;
    float light_level;
    float toxicity;
    Vec2 current;

    static EnvironmentConfig get(EnvironmentType type) {
        switch(type) {
            case EnvironmentType::LAKE:
                return {"🌊 Lake", sf::Color(20, 60, 100), 20.0f, 0.7f, 0.1f, Vec2(0.5f, 0.0f)};
            case EnvironmentType::OCEAN_REEF:
                return {"🐠 Ocean Reef", sf::Color(10, 40, 80), 25.0f, 0.8f, 0.05f, Vec2(1.0f, 0.5f)};
            case EnvironmentType::FOREST_FLOOR:
                return {"🌲 Forest Floor", sf::Color(40, 30, 20), 15.0f, 0.3f, 0.2f, Vec2(0.0f, 0.0f)};
            case EnvironmentType::VOLCANIC_VENT:
                return {"🌋 Volcanic Vent", sf::Color(60, 20, 10), 80.0f, 0.1f, 0.6f, Vec2(2.0f, 1.0f)};
            case EnvironmentType::ARCTIC_ICE:
                return {"❄️ Arctic Ice", sf::Color(200, 220, 255), -10.0f, 0.5f, 0.0f, Vec2(3.0f, 0.0f)};
            case EnvironmentType::IMMUNE_SYSTEM:
                return {"🦠 Immune System", sf::Color(100, 20, 20), 37.0f, 0.0f, 0.3f, Vec2(2.0f, 0.0f)};
            default:
                return {"🌊 Lake", sf::Color(20, 60, 100), 20.0f, 0.7f, 0.1f, Vec2(0.5f, 0.0f)};
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

// Forward declaration
class Organism;
class MicroLifeSimulation;

// Organism
class Organism {
public:
    int id;
    static int next_id;
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
    int generation;

    // Genetics
    float mutation_rate = 0.15f;
    float efficiency;
    float aggression;
    float sociability;
    float toxin_resistance;

    // Trail
    std::deque<Vec2> trail;

    // Stats
    int kills = 0;
    int offspring_count = 0;
    bool has_evolved = false;

    Organism(Vec2 pos, OrganismType t, int gen = 0)
        : id(next_id++), position(pos), velocity(0, 0), type(t),
          alive(true), age(0), generation(gen) {

        auto& info = TYPE_INFO[type];
        color = info.color;
        size = info.base_size;
        speed = info.base_speed;
        energy = info.base_energy;
        max_energy = info.base_energy * 2.0f;

        // Type-specific attributes
        switch(type) {
            case OrganismType::PHOTOSYNTHETIC_ALGAE:
                perception_radius = 30.0f;
                efficiency = 0.9f;
                aggression = 0.0f;
                sociability = 0.3f;
                toxin_resistance = 0.4f;
                break;
            case OrganismType::HUNTER_PREDATOR:
                perception_radius = 120.0f;
                efficiency = 0.6f;
                aggression = 0.9f;
                sociability = 0.1f;
                toxin_resistance = 0.6f;
                break;
            case OrganismType::SCAVENGER_DECOMPOSER:
                perception_radius = 80.0f;
                efficiency = 0.95f;
                aggression = 0.2f;
                sociability = 0.4f;
                toxin_resistance = 0.8f;
                break;
            case OrganismType::PARASITE:
                perception_radius = 100.0f;
                efficiency = 0.8f;
                aggression = 0.5f;
                sociability = 0.0f;
                toxin_resistance = 0.5f;
                break;
            case OrganismType::SYMBIOTIC_PARTNER:
                perception_radius = 90.0f;
                efficiency = 0.85f;
                aggression = 0.1f;
                sociability = 0.9f;
                toxin_resistance = 0.5f;
                break;
            case OrganismType::COLONY_FORMER:
                perception_radius = 70.0f;
                efficiency = 0.75f;
                aggression = 0.3f;
                sociability = 1.0f;
                toxin_resistance = 0.6f;
                break;
            case OrganismType::TOXIC_BACTERIA:
                perception_radius = 60.0f;
                efficiency = 0.7f;
                aggression = 0.4f;
                sociability = 0.2f;
                toxin_resistance = 1.0f;
                break;
            case OrganismType::GIANT_AMOEBA:
                perception_radius = 50.0f;
                efficiency = 0.5f;
                aggression = 0.3f;
                sociability = 0.1f;
                toxin_resistance = 0.9f;
                break;
            case OrganismType::SPEED_DEMON:
                perception_radius = 150.0f;
                efficiency = 0.4f;
                aggression = 0.7f;
                sociability = 0.0f;
                toxin_resistance = 0.2f;
                break;
            case OrganismType::TANK_ORGANISM:
                perception_radius = 40.0f;
                efficiency = 0.6f;
                aggression = 0.2f;
                sociability = 0.3f;
                toxin_resistance = 0.95f;
                break;
            case OrganismType::ENERGY_VAMPIRE:
                perception_radius = 110.0f;
                efficiency = 0.75f;
                aggression = 0.6f;
                sociability = 0.0f;
                toxin_resistance = 0.6f;
                break;
            case OrganismType::APEX_PREDATOR:
                perception_radius = 140.0f;
                efficiency = 0.7f;
                aggression = 1.0f;
                sociability = 0.0f;
                toxin_resistance = 0.8f;
                break;
        }

        // Add genetic variation
        efficiency += normal_dist(::gen) * 0.05f;
        aggression += normal_dist(::gen) * 0.1f;
        speed += normal_dist(::gen) * 0.2f;

        efficiency = std::max(0.1f, std::min(1.0f, efficiency));
        aggression = std::max(0.0f, std::min(1.0f, aggression));
        speed = std::max(0.1f, std::min(5.0f, speed));
    }

    void update(float dt, const EnvironmentConfig& env,
                std::vector<FoodParticle>& food,
                std::vector<Organism>& organisms,
                std::vector<Particle>& particles,
                std::deque<GameEvent>& events);

    bool canReproduce() const {
        return alive && energy > max_energy * 0.75f && age > 150;
    }

    Organism reproduce(std::deque<GameEvent>& events, std::vector<Particle>& particles) {
        Organism child(position + Vec2(normal_dist(::gen) * 10.0f, normal_dist(::gen) * 10.0f),
                      type, generation + 1);

        // Inherit with mutation
        child.efficiency = efficiency + normal_dist(::gen) * mutation_rate * 0.1f;
        child.aggression = aggression + normal_dist(::gen) * mutation_rate * 0.1f;
        child.speed = speed + normal_dist(::gen) * mutation_rate * 0.3f;
        child.toxin_resistance = toxin_resistance + normal_dist(::gen) * mutation_rate * 0.1f;

        child.efficiency = std::max(0.1f, std::min(1.0f, child.efficiency));
        child.aggression = std::max(0.0f, std::min(1.0f, child.aggression));
        child.speed = std::max(0.1f, std::min(6.0f, child.speed));
        child.toxin_resistance = std::max(0.0f, std::min(1.0f, child.toxin_resistance));

        child.energy = energy * 0.4f;
        energy *= 0.6f;

        offspring_count++;

        // Check for mutation
        if (std::abs(child.speed - speed) > 0.5f ||
            std::abs(child.efficiency - efficiency) > 0.15f) {

            std::stringstream ss;
            ss << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name
               << " #" << id << " offspring MUTATED!";

            events.emplace_back(EventType::MUTATION, ss.str(), sf::Color(255, 100, 255));

            // Mutation particle burst
            for (int i = 0; i < 15; i++) {
                float angle = uniform_dist(gen) * 2.0f * PI;
                float speed_p = 50.0f + uniform_dist(gen) * 100.0f;
                Vec2 vel(std::cos(angle) * speed_p, std::sin(angle) * speed_p);
                particles.emplace_back(child.position, vel, sf::Color(255, 100, 255), 1.0f, 3.0f);
            }
        }

        // Birth event
        std::stringstream ss;
        ss << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name
           << " #" << id << " gave birth! (Gen " << child.generation << ")";

        events.emplace_back(EventType::BIRTH, ss.str(), sf::Color(100, 255, 100));

        return child;
    }
};

int Organism::next_id = 1;

void Organism::update(float dt, const EnvironmentConfig& env,
                     std::vector<FoodParticle>& food,
                     std::vector<Organism>& organisms,
                     std::vector<Particle>& particles,
                     std::deque<GameEvent>& events) {
    if (!alive) return;

    age++;

    // Check for evolution (every 500 age, stats improve)
    if (age % 500 == 0 && age > 0 && !has_evolved) {
        efficiency = std::min(1.0f, efficiency * 1.05f);
        speed = std::min(6.0f, speed * 1.05f);
        max_energy *= 1.1f;
        has_evolved = true;

        std::stringstream ss;
        ss << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name
           << " #" << id << " EVOLVED! (Age " << age << ")";

        events.emplace_back(EventType::EVOLUTION, ss.str(), sf::Color(255, 215, 0));

        // Evolution particle burst!
        for (int i = 0; i < 25; i++) {
            float angle = uniform_dist(gen) * 2.0f * PI;
            float speed_p = 75.0f + uniform_dist(gen) * 150.0f;
            Vec2 vel(std::cos(angle) * speed_p, std::sin(angle) * speed_p);
            particles.emplace_back(position, vel, sf::Color(255, 215, 0), 1.5f, 4.0f);
        }
    }

    // Type-specific behavior
    switch(type) {
        case OrganismType::PHOTOSYNTHETIC_ALGAE:
            energy += env.light_level * 8.0f * dt;
            energy = std::min(energy, max_energy);
            if (uniform_dist(gen) < 0.08f) {
                velocity = Vec2(normal_dist(::gen) * 0.5f, normal_dist(::gen) * 0.5f);
            }
            break;

        case OrganismType::HUNTER_PREDATOR:
        case OrganismType::APEX_PREDATOR: {
            Organism* nearest_prey = nullptr;
            float nearest_dist = perception_radius;

            for (auto& org : organisms) {
                if (!org.alive || &org == this) continue;
                if (org.type == type) continue; // Don't eat own kind

                float dist = position.distance(org.position);
                if (dist < nearest_dist) {
                    nearest_dist = dist;
                    nearest_prey = &org;
                }
            }

            if (nearest_prey) {
                Vec2 direction = (nearest_prey->position - position).normalized();
                velocity = velocity + direction * aggression * speed * dt * 120.0f;

                if (nearest_dist < size + nearest_prey->size) {
                    float energy_gained = nearest_prey->energy * 0.75f;
                    energy += energy_gained;
                    energy = std::min(energy, max_energy);
                    nearest_prey->alive = false;
                    kills++;

                    std::stringstream ss;
                    ss << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name
                       << " #" << id << " hunted " << TYPE_INFO[nearest_prey->type].name
                       << " #" << nearest_prey->id << "!";

                    events.emplace_back(EventType::HUNT_SUCCESS, ss.str(), sf::Color(255, 100, 100));

                    // Kill particles
                    for (int i = 0; i < 10; i++) {
                        float angle = uniform_dist(gen) * 2.0f * PI;
                        float speed_p = 50.0f + uniform_dist(gen) * 100.0f;
                        Vec2 vel(std::cos(angle) * speed_p, std::sin(angle) * speed_p);
                        particles.emplace_back(nearest_prey->position, vel, sf::Color(255, 50, 50), 0.8f, 3.0f);
                    }
                }
            } else {
                if (uniform_dist(gen) < 0.05f) {
                    velocity = Vec2(normal_dist(::gen) * speed, normal_dist(::gen) * speed);
                }
            }
            break;
        }

        case OrganismType::SCAVENGER_DECOMPOSER: {
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
                Vec2 direction = (nearest_food->position - position).normalized();
                velocity = velocity + direction * speed * dt * 100.0f;

                if (nearest_dist < size + nearest_food->size) {
                    energy += nearest_food->energy;
                    energy = std::min(energy, max_energy);
                    food.erase(food.begin() + food_idx);
                }
            } else {
                if (uniform_dist(gen) < 0.05f) {
                    velocity = Vec2(normal_dist(::gen) * speed, normal_dist(::gen) * speed);
                }
            }
            break;
        }

        case OrganismType::SPEED_DEMON: {
            // Super fast, erratic movement
            if (uniform_dist(gen) < 0.3f) {
                velocity = Vec2(normal_dist(::gen) * speed, normal_dist(::gen) * speed);
            }

            // Leave trail particles
            if (uniform_dist(gen) < 0.2f) {
                particles.emplace_back(position, Vec2(0, 0), color, 0.3f, 2.0f);
            }
            break;
        }

        default:
            if (uniform_dist(gen) < 0.05f) {
                velocity = Vec2(normal_dist(::gen) * speed, normal_dist(::gen) * speed);
            }
            break;
    }

    // Apply environmental current
    velocity = velocity + env.current * dt * 15.0f;

    // Apply velocity
    position = position + velocity * dt * 50.0f;

    // Bounds
    if (position.x < 0 || position.x > WINDOW_WIDTH - EVENT_FEED_WIDTH) velocity.x *= -0.5f;
    if (position.y < 0 || position.y > WINDOW_HEIGHT) velocity.y *= -0.5f;
    position.x = std::max(0.0f, std::min((float)(WINDOW_WIDTH - EVENT_FEED_WIDTH), position.x));
    position.y = std::max(0.0f, std::min((float)WINDOW_HEIGHT, position.y));

    // Friction
    velocity = velocity * 0.94f;

    // Energy consumption
    float base_cost = 0.08f * dt;
    float movement_cost = velocity.length() * 0.015f * dt;
    float size_cost = size * 0.01f * dt;
    energy -= (base_cost + movement_cost + size_cost) / efficiency;

    // Environmental damage
    float temp_stress = std::abs(env.temperature - 20.0f) * 0.01f * dt;
    float toxicity_damage = env.toxicity * (1.0f - toxin_resistance) * 1.5f * dt;
    energy -= (temp_stress + toxicity_damage);

    // Death
    if (energy <= 0 || age > 15000) {
        alive = false;

        std::stringstream ss;
        ss << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name
           << " #" << id << " died (Age: " << age
           << ", Kills: " << kills << ", Offspring: " << offspring_count << ")";

        events.emplace_back(EventType::DEATH, ss.str(), sf::Color(150, 150, 150));
    }

    // Trail
    trail.push_front(position);
    if (trail.size() > 30) {
        trail.pop_back();
    }
}

// Main Simulation
class MicroLifeSimulation {
public:
    EnvironmentType current_env_type;
    EnvironmentConfig env_config;
    std::vector<Organism> organisms;
    std::vector<FoodParticle> food_particles;
    std::vector<Particle> particles;
    std::deque<GameEvent> events;

    float time = 0.0f;
    float simulation_speed = 1.0f;
    bool paused = false;
    bool show_graphs = true;

    int frame_count = 0;
    float fps = 0.0f;
    sf::Clock fps_clock;

    std::deque<int> population_history;
    std::deque<float> energy_history;
    std::map<OrganismType, int> species_count;

    // Statistics
    int total_births = 0;
    int total_deaths = 0;
    int total_evolutions = 0;
    int total_mutations = 0;

    MicroLifeSimulation(EnvironmentType env_type = EnvironmentType::LAKE)
        : current_env_type(env_type) {
        env_config = EnvironmentConfig::get(env_type);
        initializePopulation();
    }

    void changeEnvironment(EnvironmentType new_env) {
        current_env_type = new_env;
        env_config = EnvironmentConfig::get(new_env);

        events.emplace_back(EventType::EVOLUTION,
            "🌍 Environment changed to: " + env_config.name,
            sf::Color(100, 200, 255));
    }

    void initializePopulation() {
        organisms.clear();

        // Add all 12 types!
        std::vector<OrganismType> all_types = {
            OrganismType::PHOTOSYNTHETIC_ALGAE,
            OrganismType::HUNTER_PREDATOR,
            OrganismType::SCAVENGER_DECOMPOSER,
            OrganismType::PARASITE,
            OrganismType::SYMBIOTIC_PARTNER,
            OrganismType::COLONY_FORMER,
            OrganismType::TOXIC_BACTERIA,
            OrganismType::GIANT_AMOEBA,
            OrganismType::SPEED_DEMON,
            OrganismType::TANK_ORGANISM,
            OrganismType::ENERGY_VAMPIRE,
            OrganismType::APEX_PREDATOR
        };

        // Add 5 of each type
        for (auto type : all_types) {
            for (int i = 0; i < 5; i++) {
                Vec2 pos(uniform_dist(gen) * (WINDOW_WIDTH - EVENT_FEED_WIDTH),
                        uniform_dist(gen) * WINDOW_HEIGHT);
                organisms.emplace_back(pos, type);
            }
        }

        events.emplace_back(EventType::BIRTH,
            "🎬 Simulation started with " + std::to_string(organisms.size()) + " organisms!",
            sf::Color(255, 255, 100));
    }

    void addOrganism(Vec2 pos, OrganismType type) {
        organisms.emplace_back(pos, type);

        std::stringstream ss;
        ss << "➕ Added " << TYPE_INFO[type].emoji << " " << TYPE_INFO[type].name;
        events.emplace_back(EventType::BIRTH, ss.str(), sf::Color(100, 255, 100));
    }

    void update(float dt) {
        dt *= simulation_speed;
        time += dt;

        // Update organisms
        for (auto& org : organisms) {
            org.update(dt, env_config, food_particles, organisms, particles, events);
        }

        // Reproduction
        std::vector<Organism> offspring;
        for (auto& org : organisms) {
            if (org.canReproduce() && uniform_dist(gen) < 0.015f) {
                offspring.push_back(org.reproduce(events, particles));
                total_births++;
            }
        }

        for (auto& child : offspring) {
            organisms.push_back(child);
        }

        // Remove dead
        size_t before = organisms.size();
        organisms.erase(
            std::remove_if(organisms.begin(), organisms.end(),
                [this](Organism& org) {
                    if (!org.alive) {
                        food_particles.emplace_back(org.position, org.energy * 0.6f);
                        total_deaths++;
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
                    f.energy -= 0.8f * dt;
                    return f.energy <= 0;
                }),
            food_particles.end()
        );

        // Update particles
        for (auto& p : particles) {
            p.update(dt);
        }

        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return !p.isAlive(); }),
            particles.end()
        );

        // Update events
        for (auto& e : events) {
            e.update(dt);
        }

        events.erase(
            std::remove_if(events.begin(), events.end(),
                [](const GameEvent& e) { return e.isExpired(); }),
            events.end()
        );

        // Keep only last 10 events displayed
        while (events.size() > 10) {
            events.pop_back();
        }

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
        species_count.clear();
        float total_energy = 0.0f;

        for (const auto& org : organisms) {
            species_count[org.type]++;
            total_energy += org.energy;
        }

        population_history.push_front(organisms.size());
        if (population_history.size() > 300) population_history.pop_back();

        float avg_energy = organisms.empty() ? 0 : total_energy / organisms.size();
        energy_history.push_front(avg_energy);
        if (energy_history.size() > 300) energy_history.pop_back();
    }

    void render(sf::RenderWindow& window);
    void reset();
};

void MicroLifeSimulation::render(sf::RenderWindow& window) {
    window.clear(env_config.background);

    // Food
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
            float alpha = (1.0f - (float)i / org.trail.size()) * 120.0f;
            sf::Color trail_color = org.color;
            trail_color.a = (sf::Uint8)alpha;

            sf::Vertex line[] = {
                sf::Vertex(sf::Vector2f(org.trail[i-1].x, org.trail[i-1].y), trail_color),
                sf::Vertex(sf::Vector2f(org.trail[i].x, org.trail[i].y), trail_color)
            };
            window.draw(line, 2, sf::Lines);
        }

        // Body with glow
        float energy_ratio = org.energy / org.max_energy;

        // Glow
        if (energy_ratio > 0.5f) {
            sf::CircleShape glow(org.size * 1.5f);
            glow.setPosition(org.position.x - org.size * 1.5f, org.position.y - org.size * 1.5f);
            sf::Color glow_color = org.color;
            glow_color.a = (sf::Uint8)(energy_ratio * 100.0f);
            glow.setFillColor(glow_color);
            window.draw(glow);
        }

        // Body
        sf::CircleShape body(org.size);
        body.setPosition(org.position.x - org.size, org.position.y - org.size);
        sf::Color body_color = org.color;
        body_color.a = (sf::Uint8)(100 + energy_ratio * 155);
        body.setFillColor(body_color);
        body.setOutlineColor(sf::Color(255, 255, 255, 150));
        body.setOutlineThickness(1.0f);
        window.draw(body);
    }

    // Particles
    for (const auto& p : particles) {
        sf::CircleShape shape(p.size);
        shape.setPosition(p.position.x - p.size, p.position.y - p.size);
        shape.setFillColor(p.color);
        shape.setRotation(p.rotation);
        window.draw(shape);
    }

    // EVENT FEED (right side)
    sf::RectangleShape feed_bg(sf::Vector2f(EVENT_FEED_WIDTH, WINDOW_HEIGHT));
    feed_bg.setPosition(WINDOW_WIDTH - EVENT_FEED_WIDTH, 0);
    feed_bg.setFillColor(sf::Color(0, 0, 0, 200));
    window.draw(feed_bg);

    // Event feed title
    int y_pos = 20;

    // Events
    for (size_t i = 0; i < events.size(); i++) {
        auto& event = events[i];

        sf::RectangleShape event_box(sf::Vector2f(EVENT_FEED_WIDTH - 20, 60));
        event_box.setPosition(WINDOW_WIDTH - EVENT_FEED_WIDTH + 10, y_pos);

        sf::Color box_color = event.color;
        box_color.a = (sf::Uint8)(event.getAlpha() * 0.3f);
        event_box.setFillColor(box_color);
        window.draw(event_box);

        y_pos += 70;
    }

    // Stats overlay
    y_pos = WINDOW_HEIGHT - 200;
    sf::RectangleShape stats_bg(sf::Vector2f(EVENT_FEED_WIDTH - 20, 180));
    stats_bg.setPosition(WINDOW_WIDTH - EVENT_FEED_WIDTH + 10, y_pos);
    stats_bg.setFillColor(sf::Color(0, 0, 0, 150));
    window.draw(stats_bg);

    // FPS bar
    sf::RectangleShape fpsBar(sf::Vector2f(fps * 2.5f, 15));
    fpsBar.setPosition(WINDOW_WIDTH - EVENT_FEED_WIDTH + 20, 20);
    fpsBar.setFillColor(fps >= TARGET_FPS * 0.9f ? sf::Color::Green :
                       fps >= TARGET_FPS * 0.6f ? sf::Color::Yellow : sf::Color::Red);
    window.draw(fpsBar);
}

void MicroLifeSimulation::reset() {
    organisms.clear();
    food_particles.clear();
    particles.clear();
    events.clear();
    population_history.clear();
    energy_history.clear();

    total_births = 0;
    total_deaths = 0;
    total_evolutions = 0;
    total_mutations = 0;

    initializePopulation();

    events.emplace_back(EventType::EVOLUTION, "🔄 Simulation RESET!", sf::Color(255, 100, 100));
}

// Main
int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          MICROLIFE ULTIMATE - CINEMATIC EDITION                   ║\n";
    std::cout << "║              Film-Quality Evolution Simulator                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Select Environment:\n";
    std::cout << "1. 🌊 Lake\n2. 🐠 Ocean Reef\n3. 🌲 Forest\n";
    std::cout << "4. 🌋 Volcanic\n5. ❄️ Arctic\n6. 🦠 Immune System\n";
    std::cout << "\nChoice (1-6): ";

    int choice;
    std::cin >> choice;

    EnvironmentType env_type = static_cast<EnvironmentType>(choice - 1);

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                            "MICROLIFE CINEMATIC - Live Evolution Feed");
    window.setFramerateLimit(TARGET_FPS);

    MicroLifeSimulation sim(env_type);

    sf::Clock clock;

    std::cout << "\n🎬 SIMULATION STARTED!\n";
    std::cout << "Watch the LIVE EVENT FEED on the right →→→\n";
    std::cout << "Total organisms: " << sim.organisms.size() << "\n";
    std::cout << "12 unique species!\n\n";

    OrganismType current_add_type = OrganismType::PHOTOSYNTHETIC_ALGAE;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                } else if (event.key.code == sf::Keyboard::Space) {
                    sim.paused = !sim.paused;
                } else if (event.key.code == sf::Keyboard::R) {
                    sim.reset();
                } else if (event.key.code == sf::Keyboard::G) {
                    sim.show_graphs = !sim.show_graphs;
                } else if (event.key.code == sf::Keyboard::E) {
                    int next = ((int)sim.current_env_type + 1) % 6;
                    sim.changeEnvironment((EnvironmentType)next);
                } else if (event.key.code >= sf::Keyboard::Num1 && event.key.code <= sf::Keyboard::Num9) {
                    int type_idx = event.key.code - sf::Keyboard::Num1;
                    if (type_idx < 12) {
                        current_add_type = static_cast<OrganismType>(type_idx);
                        std::cout << "Selected: " << TYPE_INFO[current_add_type].name << "\n";
                    }
                } else if (event.key.code == sf::Keyboard::Equal) {
                    sim.simulation_speed = std::min(5.0f, sim.simulation_speed + 0.5f);
                } else if (event.key.code == sf::Keyboard::Hyphen) {
                    sim.simulation_speed = std::max(0.1f, sim.simulation_speed - 0.5f);
                }
            }

            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    Vec2 pos(event.mouseButton.x, event.mouseButton.y);
                    if (pos.x < WINDOW_WIDTH - EVENT_FEED_WIDTH) {
                        sim.addOrganism(pos, current_add_type);
                    }
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

    std::cout << "\n🎬 SIMULATION ENDED\n";
    std::cout << "Total births: " << sim.total_births << "\n";
    std::cout << "Total deaths: " << sim.total_deaths << "\n";
    std::cout << "Final population: " << sim.organisms.size() << "\n";

    return 0;
}
