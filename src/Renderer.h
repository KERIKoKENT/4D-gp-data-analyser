#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <glm/glm.hpp>
#include "Camera.h"
#include "Grid.h"

class Renderer {
public:
    Renderer(int width, int height, const char* title);
    ~Renderer();

    void Time();
    void Clear();
    void SwapBuffers();
    bool ShouldClose();
    void PollEvents();

    void SetupGrid(const Grid& grid);
    void DrawGrid(const Grid& grid);
    void UpdateGridData(float* vertices, int sizeInBytes);
    void RecreateGrid(const Grid& grid);

    GLFWwindow* GetWindow() const { return window; }
    Camera* GetCamera() const { return camera; }
    double GetDeltaTime() const { return deltaTime; }

private:
    GLFWwindow* window;
    int screenWidth, screenHeight;
    double lastTime, currentTime, deltaTime;

    Camera* camera;
    std::unordered_map<std::string, GLuint> shaderPrograms;

    GLuint gridVAO = 0, gridVBO = 0, gridEBO = 0;
    int gridIndexCount = 0;

    void InitShaders();
    void ProcessInput(GLFWwindow* window, float deltaTime);

    std::string ReadFile(const char* filePath);
    GLuint CompileShader(GLenum shaderType, const std::string& shaderSource);
    GLuint CreateProgram(GLuint vertexShader, GLuint fragmentShader);
    std::vector<std::string> GetShaderFiles(const std::string& directory);
};

#endif
