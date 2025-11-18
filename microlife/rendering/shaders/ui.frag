#version 330 core

in vec2 frag_texcoord;
in vec4 frag_color;

out vec4 out_color;

uniform sampler2D font_texture;
uniform bool use_texture;

void main() {
    if (use_texture) {
        // Text rendering with font atlas
        float alpha = texture(font_texture, frag_texcoord).r;
        out_color = vec4(frag_color.rgb, frag_color.a * alpha);
    } else {
        // Solid color (for rectangles, lines, etc.)
        out_color = frag_color;
    }
}
