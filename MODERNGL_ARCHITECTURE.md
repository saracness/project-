# ModernGL Production Renderer Architecture
**Version:** 1.0
**Target:** 100+ FPS, 10,000+ organisms
**Quality:** Enterprise/Production-grade

---

## 🎯 Design Goals

1. **Performance:** 100+ FPS with 1000 organisms, 60+ FPS with 10,000
2. **Quality:** AAA-game level graphics
3. **Maintainability:** Clean, documented, testable code
4. **Flexibility:** Easy to extend and modify
5. **Compatibility:** Cross-platform (Linux, Windows, Mac)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│              (Python Simulation Logic)                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              GLRenderer (Main Renderer)                  │
│  - Window management                                     │
│  - Rendering pipeline                                    │
│  - Resource management                                   │
└────────┬────────────┬────────────┬──────────────────────┘
         │            │            │
    ┌────▼───┐  ┌────▼───┐  ┌────▼────┐
    │Shaders │  │Buffers │  │Textures │
    └────┬───┘  └────┬───┘  └────┬────┘
         │            │            │
    ┌────▼────────────▼────────────▼────┐
    │         OpenGL Context             │
    │         (ModernGL)                 │
    └────────────────┬───────────────────┘
                     │
                ┌────▼────┐
                │   GPU   │
                └─────────┘
```

---

## 📦 Component Structure

### 1. Core Renderer (`microlife/rendering/gl_renderer.py`)

**Responsibilities:**
- Window creation (moderngl-window)
- OpenGL context management
- Main rendering loop
- Resource lifecycle
- Performance monitoring

**Key Classes:**
```python
class GLRenderer:
    - __init__(width, height, title)
    - setup()           # Initialize OpenGL resources
    - render(sim_data)  # Main render call
    - cleanup()         # Resource cleanup
    - get_stats()       # Performance statistics
```

### 2. Shader Manager (`microlife/rendering/shaders/shader_manager.py`)

**Responsibilities:**
- Shader compilation
- Program linking
- Uniform management
- Shader hot-reloading (development)

**Key Classes:**
```python
class ShaderProgram:
    - compile(vertex_src, fragment_src)
    - set_uniform(name, value)
    - use()

class ShaderManager:
    - load_shader(name)
    - get_shader(name)
    - reload_all()  # Hot reload for development
```

### 3. Organism Renderer (`microlife/rendering/organism_renderer.py`)

**Responsibilities:**
- Instanced rendering (draw 10,000+ organisms in one call)
- Morphology-based visual representation
- Glow effects for AI organisms
- Smooth interpolation

**Technique:** Instanced Rendering
```glsl
// Vertex Shader
#version 330 core
layout(location = 0) in vec2 position;      // Circle vertex
layout(location = 1) in vec2 instance_pos;  // Organism position
layout(location = 2) in vec3 instance_color;
layout(location = 3) in float instance_size;
layout(location = 4) in float instance_glow;

void main() {
    vec2 world_pos = position * instance_size + instance_pos;
    gl_Position = projection * view * vec4(world_pos, 0.0, 1.0);
    // ...
}
```

### 4. Particle System (`microlife/rendering/particle_renderer.py`)

**Responsibilities:**
- GPU-based particle physics
- Point sprite rendering
- Particle pooling
- Effect types (food, death, reproduction)

**Technique:** Compute Shaders + Point Sprites
```glsl
// Compute Shader - Particle Physics
#version 430 core
layout(local_size_x = 256) in;

struct Particle {
    vec2 position;
    vec2 velocity;
    vec4 color;
    float lifetime;
    float size;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= particle_count) return;

    // Update physics on GPU
    particles[id].velocity.y -= gravity * dt;
    particles[id].position += particles[id].velocity * dt;
    particles[id].lifetime -= dt;
    // ...
}
```

### 5. Trail System (`microlife/rendering/trail_renderer.py`)

**Responsibilities:**
- Vertex buffer based trails
- Fade-out effect
- Batch rendering
- Circular buffer for positions

**Technique:** Dynamic Vertex Buffers
```python
# Trail data structure
trail_data = {
    'positions': [],  # List of vec2
    'colors': [],     # List of vec4 (with alpha fade)
    'indices': []     # Line strip indices
}

# Update trail buffer each frame
vbo.write(trail_data)
```

### 6. Heatmap Generator (`microlife/rendering/heatmap_renderer.py`)

**Responsibilities:**
- Density calculation on GPU
- Gaussian blur (compute shader)
- Color gradient mapping
- Transparent overlay

**Technique:** Compute Shaders
```glsl
// Compute Shader - Density Calculation
#version 430 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f, binding = 0) uniform image2D density_map;

uniform vec2 organism_positions[MAX_ORGANISMS];
uniform int organism_count;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec2 pixel_pos = vec2(pixel) / resolution;

    float density = 0.0;
    for (int i = 0; i < organism_count; i++) {
        float dist = distance(pixel_pos, organism_positions[i]);
        density += exp(-dist * dist / (2.0 * sigma * sigma));
    }

    imageStore(density_map, pixel, vec4(density));
}
```

### 7. UI Overlay (`microlife/rendering/ui_overlay.py`)

**Responsibilities:**
- FPS counter
- Statistics display
- Controls hint
- Performance graphs

**Technique:** Text rendering + ImGui-style layout
```python
class UIOverlay:
    - render_fps(fps)
    - render_stats(organisms, particles, etc)
    - render_graphs(frame_times)
    - render_controls()
```

### 8. Camera System (`microlife/rendering/camera.py`)

**Responsibilities:**
- View matrix management
- Zoom/pan controls
- Smooth interpolation
- Bounds checking

```python
class Camera:
    - position: vec2
    - zoom: float
    - update(dt)
    - screen_to_world(screen_pos)
    - world_to_screen(world_pos)
    - get_view_matrix()
    - get_projection_matrix()
```

---

## 🎨 Rendering Pipeline

### Frame Rendering Order:

```
1. Clear framebuffer
2. Update camera matrices
3. Render heatmap (if enabled)
   └─ Compute density
   └─ Apply blur
   └─ Render as textured quad
4. Render trails
   └─ Update trail buffers
   └─ Draw line strips with fade
5. Render organisms
   └─ Update instance buffer
   └─ Instanced draw call
6. Render particles
   └─ Update particle buffer (compute shader)
   └─ Draw point sprites
7. Render UI overlay
   └─ FPS counter
   └─ Statistics
   └─ Controls
8. Swap buffers
```

---

## 📊 Performance Optimizations

### 1. Instanced Rendering
```python
# Single draw call for all organisms
organisms_vbo.write(instance_data)  # CPU→GPU once per frame
ctx.draw(vao, mode=moderngl.TRIANGLE_FAN,
         vertices=32, instances=organism_count)
```

### 2. Compute Shaders
```python
# GPU-side particle physics
particle_shader.run(group_x=num_particles // 256)
# No CPU involvement!
```

### 3. Double Buffering
```python
# Ping-pong buffers for smooth updates
current_buffer = 0
buffers = [buffer_a, buffer_b]

def update():
    read_buffer = buffers[current_buffer]
    write_buffer = buffers[1 - current_buffer]
    # GPU reads from read_buffer, writes to write_buffer
    current_buffer = 1 - current_buffer
```

### 4. Batching
```python
# Batch all similar draw calls
with shader_program:
    for batch in batches:
        vao.bind()
        ctx.draw_indirect(batch.indirect_buffer)
```

### 5. Frustum Culling
```python
# Only render visible organisms
visible = [org for org in organisms
           if camera.is_visible(org.position, org.size)]
```

---

## 🗂️ File Structure

```
microlife/
├── rendering/
│   ├── __init__.py
│   ├── gl_renderer.py           # Main renderer
│   ├── camera.py                # Camera system
│   ├── organism_renderer.py     # Instanced organism rendering
│   ├── particle_renderer.py     # GPU particle system
│   ├── trail_renderer.py        # Trail rendering
│   ├── heatmap_renderer.py      # Compute shader heatmap
│   ├── ui_overlay.py            # UI rendering
│   ├── shaders/
│   │   ├── __init__.py
│   │   ├── shader_manager.py
│   │   ├── organism.vert        # Organism vertex shader
│   │   ├── organism.frag        # Organism fragment shader
│   │   ├── particle.vert
│   │   ├── particle.frag
│   │   ├── particle_update.comp # Particle physics compute
│   │   ├── trail.vert
│   │   ├── trail.frag
│   │   ├── heatmap_density.comp # Density calculation
│   │   ├── heatmap_blur.comp    # Gaussian blur
│   │   ├── heatmap.vert
│   │   ├── heatmap.frag
│   │   ├── ui.vert
│   │   └── ui.frag
│   └── resources/
│       ├── fonts/               # Font atlases
│       └── textures/            # Particle textures
├── simulation/                  # Existing simulation code
└── ml/                          # Existing AI code

demo_gl.py                       # ModernGL demo
benchmark_gl.py                  # Performance benchmark
```

---

## 📈 Performance Targets

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| FPS (100 organisms) | 120+ | 144+ |
| FPS (1,000 organisms) | 100+ | 120+ |
| FPS (10,000 organisms) | 60+ | 100+ |
| Frame time | <8ms | <6ms |
| GPU memory | <500MB | <300MB |
| Startup time | <2s | <1s |

---

## 🔧 Technology Stack

**Core:**
- `moderngl` - Modern OpenGL wrapper
- `moderngl-window` - Window management
- `numpy` - Data arrays
- `pyrr` - Matrix math (view/projection)

**Optional:**
- `Pillow` - Screenshots
- `imgui` - Advanced UI (if needed)
- `glfw` - Alternative window backend

---

## 🎯 Milestones

**Milestone 1: Basic Rendering** (Day 1)
- ✅ Window + OpenGL context
- ✅ Simple organism rendering
- ✅ Camera controls

**Milestone 2: Performance** (Day 2)
- ✅ Instanced rendering
- ✅ 1000+ organisms @ 100 FPS

**Milestone 3: Effects** (Day 2-3)
- ✅ Particle system
- ✅ Trails
- ✅ Heatmap

**Milestone 4: Polish** (Day 3)
- ✅ UI overlay
- ✅ Performance profiler
- ✅ Documentation

---

**STATUS:** Architecture Complete ✅
**NEXT:** Implementation Start
