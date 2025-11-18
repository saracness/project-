#!/usr/bin/env python3
"""
Headless FPS Benchmark - Window açmadan GPU test
"""
import moderngl
import numpy as np
import time

print("=" * 70)
print("🚀 ModernGL Headless FPS Benchmark")
print("=" * 70)
print()

# Create standalone OpenGL context (no window!)
ctx = moderngl.create_standalone_context()

print(f"✅ OpenGL Context Created")
print(f"   Version: {ctx.version_code}")
print(f"   Vendor: {ctx.info['GL_VENDOR']}")
print(f"   Renderer: {ctx.info['GL_RENDERER']}")
print()

# Create test shader
vertex_shader = """
#version 330 core
in vec2 in_position;
out vec2 v_pos;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_pos = in_position;
}
"""

fragment_shader = """
#version 330 core
in vec2 v_pos;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_pos * 0.5 + 0.5, 0.0, 1.0);
}
"""

program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader
)

print("✅ Shaders Compiled")

# Create framebuffer
fbo = ctx.framebuffer(
    color_attachments=[ctx.texture((1920, 1080), components=4)]
)

# Create vertex buffer with 10,000 circles (instanced)
num_instances = 1000

# Circle geometry (32 segments)
circle_verts = []
segments = 32
for i in range(segments + 1):
    angle = 2 * np.pi * i / segments
    circle_verts.append([np.cos(angle) * 0.01, np.sin(angle) * 0.01])

circle_verts = np.array(circle_verts, dtype=np.float32)

vbo = ctx.buffer(circle_verts.tobytes())
vao = ctx.vertex_array(program, [(vbo, '2f', 'in_position')])

print(f"✅ Created {num_instances} instances")
print()

# Benchmark
print("🏃 Running FPS Benchmark...")
print("   Rendering 1000 circles, 1000 frames")
print()

fbo.use()

frame_count = 1000
start_time = time.time()

for frame in range(frame_count):
    ctx.clear(0.05, 0.05, 0.05)
    vao.render(moderngl.TRIANGLE_FAN, instances=num_instances)

ctx.finish()  # Wait for GPU
end_time = time.time()

elapsed = end_time - start_time
fps = frame_count / elapsed
frame_time = (elapsed / frame_count) * 1000  # ms

print("=" * 70)
print("📊 BENCHMARK RESULTS")
print("=" * 70)
print(f"Total Time:      {elapsed:.2f}s")
print(f"Frames:          {frame_count}")
print(f"Average FPS:     {fps:.1f}")
print(f"Frame Time:      {frame_time:.2f}ms")
print()

if fps >= 100:
    print("✅ SUCCESS! 100+ FPS ACHIEVED! 🎉")
elif fps >= 60:
    print("⚠️  60+ FPS (Good, but not 100+)")
else:
    print("❌ Below 60 FPS (Performance issue)")

print()
print("=" * 70)

# Cleanup
vao.release()
vbo.release()
fbo.release()
program.release()
ctx.release()

print("✅ Cleanup complete")
