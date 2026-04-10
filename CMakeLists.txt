cmake_minimum_required(VERSION 3.20)
project(GPAnalyser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows")

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(${PROJECT_NAME} GPanalyser.cpp "src/Grid.cpp" "src/Grid.h" "src/Camera.cpp" "src/Camera.h" "src/Renderer.cpp" "src/Renderer.h")

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    glfw
    imgui::imgui
    Eigen3::Eigen
    OpenGL::GL
    glad::glad
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/external/LBFGSpp/include
)

file(COPY "${CMAKE_SOURCE_DIR}/src/shaders" 
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")