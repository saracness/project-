# Makefile for C++ RL Implementations

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
LIBS = -lsfml-graphics -lsfml-window -lsfml-system -pthread

.PHONY: all clean animation

all: rl_learning_animation

rl_learning_animation: rl_learning_animation.cpp
	@echo "Compiling RL Learning Animation..."
	$(CXX) $(CXXFLAGS) rl_learning_animation.cpp $(LIBS) -o rl_learning_animation
	@echo "✓ Build complete!"
	@echo ""
	@echo "Run with: ./rl_learning_animation"
	@echo ""
	@echo "Controls:"
	@echo "  SPACE: Pause/Resume"
	@echo "  +/-: Speed up/slow down"
	@echo "  S: Skip episode"
	@echo "  R: Reset learning"
	@echo "  Q/ESC: Quit"
	@echo ""

animation: rl_learning_animation
	@echo "Starting animation..."
	./rl_learning_animation

clean:
	rm -f rl_learning_animation
	@echo "Cleaned."
