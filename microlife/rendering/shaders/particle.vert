#version 330 core

// Vertex attributes (point sprite)
layout(location = 0) in vec2 particle_position;
layout(location = 1) in vec4 particle_color;  // RGBA
layout(location = 2) in float particle_size;
layout(location = 3) in float particle_lifetime_ratio;  // 0-1 (1=just spawned, 0=dead)

// Uniforms
uniform mat4 projection;
uniform mat4 view;

// Outputs
out vec4 frag_color;
out float frag_lifetime;

void main() {
    gl_Position = projection * view * vec4(particle_position, 0.0, 1.0);
    gl_PointSize = particle_size * particle_lifetime_ratio;  // Shrink over time

    frag_color = particle_color;
    frag_lifetime = particle_lifetime_ratio;
}
