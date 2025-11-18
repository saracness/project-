#version 330 core

in vec4 frag_color;
in float frag_lifetime;

out vec4 out_color;

void main() {
    // Point sprite UV coordinates (gl_PointCoord is automatic for point sprites)
    vec2 uv = gl_PointCoord * 2.0 - 1.0;  // [-1, 1]

    // Circular shape with smooth edges
    float dist = length(uv);
    if (dist > 1.0) {
        discard;
    }

    // Soft particle edge
    float alpha = (1.0 - dist) * frag_color.a * frag_lifetime;

    // Brighter center
    float brightness = 1.0 - dist * 0.5;

    out_color = vec4(frag_color.rgb * brightness, alpha);
}
