cmake_minimum_required(VERSION 3.10)
project(NeuronVisualization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system QUIET)

if(SFML_FOUND)
    message(STATUS "SFML found, building visualization")

    # Neuron visualizer (data viewer)
    add_executable(neuron_visualizer neuron_visualizer.cpp)
    target_link_libraries(neuron_visualizer sfml-graphics sfml-window sfml-system)

    # Fast neuron learning (complete simulation)
    add_executable(neuron_learning_fast neuron_learning_fast.cpp)
    target_link_libraries(neuron_learning_fast sfml-graphics sfml-window sfml-system pthread)

    # Install
    install(TARGETS neuron_visualizer neuron_learning_fast DESTINATION bin)

    message(STATUS "Building:")
    message(STATUS "  - neuron_visualizer      (data viewer)")
    message(STATUS "  - neuron_learning_fast   (fast simulation)")
else()
    message(WARNING "SFML not found, skipping visualization build")
    message(STATUS "Install SFML with: sudo apt-get install libsfml-dev (Ubuntu/Debian)")
    message(STATUS "Or: brew install sfml (macOS)")
endif()
