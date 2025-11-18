cmake_minimum_required(VERSION 3.10)
project(NeuronVisualization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system QUIET)

if(SFML_FOUND)
    message(STATUS "SFML found, building visualization")

    # Neuron visualizer executable
    add_executable(neuron_visualizer neuron_visualizer.cpp)
    target_link_libraries(neuron_visualizer sfml-graphics sfml-window sfml-system)

    # Install
    install(TARGETS neuron_visualizer DESTINATION bin)
else()
    message(WARNING "SFML not found, skipping visualization build")
    message(STATUS "Install SFML with: sudo apt-get install libsfml-dev (Ubuntu/Debian)")
    message(STATUS "Or: brew install sfml (macOS)")
endif()
