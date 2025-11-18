#version 330 core

in vec2 frag_texcoord;
out vec4 out_color;

uniform sampler2D density_texture;
uniform float alpha;  // Overall transparency

// Color gradient (blue -> green -> yellow -> red)
vec3 heatmap_color(float value) {
    // Clamp to [0, 1]
    value = clamp(value, 0.0, 1.0);

    vec3 color;
    if (value < 0.25) {
        // Blue to Cyan
        float t = value / 0.25;
        color = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 1.0), t);
    } else if (value < 0.5) {
        // Cyan to Green
        float t = (value - 0.25) / 0.25;
        color = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else if (value < 0.75) {
        // Green to Yellow
        float t = (value - 0.5) / 0.25;
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t);
    } else {
        // Yellow to Red
        float t = (value - 0.75) / 0.25;
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t);
    }

    return color;
}

void main() {
    float density = texture(density_texture, frag_texcoord).r;

    vec3 color = heatmap_color(density);

    // Alpha based on density (fade out low density areas)
    float final_alpha = alpha * smoothstep(0.0, 0.2, density);

    out_color = vec4(color, final_alpha);
}
