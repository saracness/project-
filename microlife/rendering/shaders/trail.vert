#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;  // RGBA with alpha for fade

uniform mat4 projection;
uniform mat4 view;

out vec4 frag_color;

void main() {
    gl_Position = projection * view * vec4(position, 0.0, 1.0);
    frag_color = color;
}
