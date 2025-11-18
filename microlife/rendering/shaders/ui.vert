#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec4 color;

uniform mat4 projection;

out vec2 frag_texcoord;
out vec4 frag_color;

void main() {
    gl_Position = projection * vec4(position, 0.0, 1.0);
    frag_texcoord = texcoord;
    frag_color = color;
}
