cmake_minimum_required(VERSION 3.20)
project(GPAnalyser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows")

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} GPanalyser.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    glfw
    imgui::imgui
    Eigen3::Eigen
    OpenGL::GL
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/external/LBFGSpp/include
)