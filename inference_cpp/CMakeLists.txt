cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

message(${CMAKE_INSTALL_PREFIX})

include_directories(${PROJECT_SOURCE_DIR}/include)

## OpenCV Configuration
# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories from OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# # nlohmann/json
# find_package(nlohmann_json 3.2.0 REQUIRED)


## Tensorflow Lite Configuration
# Set TFLite source directory
set(TENSORFLOW_SOURCE_DIR "/home/d568/sources/tensorflow" CACHE PATH "Directory that contains the TensorFlow project" )

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

# Create an executable
add_executable(main main.cpp utils.cpp inference.cpp detection.cpp)

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})

# Link TFLite libraries
target_link_libraries(main "tensorflow-lite")
target_link_libraries(main "xnnpack-delegate")

# Link cnpy libraries
target_link_libraries(main "cnpy")
