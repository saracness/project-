#version 330 core

// Vertex attributes
layout(location = 0) in vec2 vertex_position;  // Circle vertex (unit circle)

// Instance attributes
layout(location = 1) in vec2 instance_position;   // Organism world position
layout(location = 2) in vec3 instance_color;      // Organism color (RGB)
layout(location = 3) in float instance_size;      // Organism radius
layout(location = 4) in float instance_glow;      // Glow intensity (0-1, for AI organisms)
layout(location = 5) in float instance_energy;    // Energy level (0-1)

// Uniforms
uniform mat4 projection;
uniform mat4 view;

// Outputs to fragment shader
out vec3 frag_color;
out vec2 frag_uv;
out float frag_glow;
out float frag_energy;
out vec2 frag_center;  // For distance calculation in fragment shader

void main() {
    // Calculate world position
    vec2 world_pos = vertex_position * instance_size + instance_position;

    // Transform to clip space
    gl_Position = projection * view * vec4(world_pos, 0.0, 1.0);

    // Pass data to fragment shader
    frag_color = instance_color;
    frag_uv = vertex_position;  // UV coordinates in [-1, 1]
    frag_glow = instance_glow;
    frag_energy = instance_energy;
    frag_center = instance_position;
}
