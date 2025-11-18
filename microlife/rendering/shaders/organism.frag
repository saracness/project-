#version 330 core

// Inputs from vertex shader
in vec3 frag_color;
in vec2 frag_uv;
in float frag_glow;
in float frag_energy;
in vec2 frag_center;

// Output
out vec4 out_color;

// Uniforms
uniform float time;  // For animation effects
uniform bool enable_glow;
uniform bool enable_energy_overlay;

void main() {
    // Calculate distance from center (for circular shape)
    float dist = length(frag_uv);

    // Discard pixels outside circle
    if (dist > 1.0) {
        discard;
    }

    // Base color
    vec3 color = frag_color;

    // Energy overlay (darker when low energy)
    if (enable_energy_overlay) {
        float energy_factor = mix(0.5, 1.0, frag_energy);
        color *= energy_factor;
    }

    // Edge darkening for depth effect
    float edge = smoothstep(0.9, 1.0, dist);
    color = mix(color, color * 0.6, edge);

    // Center highlight for 3D effect
    float highlight = exp(-dist * dist * 4.0);
    color += vec3(highlight * 0.3);

    // Glow effect for AI organisms
    float glow_intensity = 0.0;
    if (enable_glow && frag_glow > 0.0) {
        // Pulsing glow animation
        float pulse = sin(time * 3.0) * 0.2 + 0.8;
        glow_intensity = frag_glow * pulse * (1.0 - dist * 0.5);

        // Add yellow glow
        vec3 glow_color = vec3(1.0, 1.0, 0.3);
        color = mix(color, glow_color, glow_intensity * 0.3);
    }

    // Smooth edge anti-aliasing
    float alpha = smoothstep(1.0, 0.98, dist);

    // Add extra alpha for glow
    if (glow_intensity > 0.0) {
        alpha = max(alpha, glow_intensity * 0.5);
    }

    out_color = vec4(color, alpha);
}
