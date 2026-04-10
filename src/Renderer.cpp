#include <fstream>
#include <sstream>
#include <filesystem>
#include <thread>
#include "Renderer.h"
#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

Renderer::Renderer(int width, int height, const char* title) {
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    screenWidth = width;
    screenHeight = height;

    window = glfwCreateWindow(screenWidth, screenHeight, title, nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        exit(EXIT_FAILURE);
    }

    glViewport(0, 0, screenWidth, screenHeight);

    glEnable(GL_DEPTH_TEST);

    camera = new Camera(glm::vec3(0.0f, 10.0f, 20.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -30.0f);

    InitShaders();
}

void Renderer::Clear() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::SwapBuffers() {
    glfwSwapBuffers(window);
}

bool Renderer::ShouldClose() {
    return glfwWindowShouldClose(window);
}

void Renderer::Time() {
    float currentFrame = (float)glfwGetTime();
    deltaTime = currentFrame - lastTime;
    lastTime = currentFrame;
}

void Renderer::PollEvents() {
    glfwPollEvents();
    Time();
    ProcessInput(window, deltaTime);
    camera->ProcessCursor(window);
}

void Renderer::SetupGrid(const Grid& grid) {
    gridIndexCount = grid.getIndicesCount(); 
    int nodes = grid.getNodesSide();         

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);
    glGenBuffers(1, &gridEBO);

    glBindVertexArray(gridVAO);

    unsigned int* indices = new unsigned int[gridIndexCount];
    grid.generateIndices(indices);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIndexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    delete[] indices;

    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);

    glBufferData(GL_ARRAY_BUFFER, grid.getNodesCount() * 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}


void Renderer::DrawGrid(const Grid& grid) {

    GLuint program = shaderPrograms["mainShader"];
    glUseProgram(program);


    float aspect = (screenHeight > 0) ? (float)screenWidth / (float)screenHeight : 1.0f;
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 1000.0f);
    glm::mat4 view = camera->GetViewMatrix();
    glm::vec3 camPos = camera->GetPosition();

    GLint projLoc = glGetUniformLocation(program, "projection");
    GLint viewLoc = glGetUniformLocation(program, "view");

    if (projLoc == -1 || viewLoc == -1) {
        std::cerr << "ERROR: Shader layout location not found! Check your .vert file names." << std::endl;
    }



    glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, glm::value_ptr(view));


    glUniform3fv(glGetUniformLocation(program, "lightPos"), 1, glm::value_ptr(camPos));
    glUniform3fv(glGetUniformLocation(program, "viewPos"), 1, glm::value_ptr(camPos));
    glUniform3f(glGetUniformLocation(program, "objectColor"), 0.2f, 0.6f, 1.0f);

    glBindVertexArray(gridVAO);
    glDrawElements(GL_TRIANGLES, gridIndexCount, GL_UNSIGNED_INT, 0);
}

Renderer::~Renderer() {
    delete camera;

    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteBuffers(1, &gridEBO);

    glfwDestroyWindow(window);
    glfwTerminate();
}

std::string Renderer::ReadFile(const char* filePath) {
    std::filesystem::path path = std::filesystem::current_path() / filePath;
    path = path.lexically_normal();
    std::ifstream file(path);
    if (!file.is_open()) {
        exit(EXIT_FAILURE);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint Renderer::CompileShader(GLenum shaderType, const std::string& shaderSource) {
    GLuint shader = glCreateShader(shaderType);
    const char* shaderCode = shaderSource.c_str();
    glShaderSource(shader, 1, &shaderCode, nullptr);
    glCompileShader(shader);
    return shader;
}

GLuint Renderer::CreateProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    if (vertexShader != 0) glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    return program;
}

std::vector<std::string> Renderer::GetShaderFiles(const std::string& directory) {
    std::vector<std::string> shaderFiles;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        shaderFiles.push_back(entry.path().string());
    }
    return shaderFiles;
}

void Renderer::InitShaders() {

    std::string path = "shaders/";

    if (!std::filesystem::exists(path)) {
        path = "../../src/shaders/";
    }

    if (!std::filesystem::exists(path)) {
        std::cerr << "CRITICAL ERROR: Shader directory not found!" << std::endl;
        std::cerr << "Current path is: " << std::filesystem::current_path() << std::endl;
        return;
    }

    std::vector<std::string> shaderFiles = GetShaderFiles(path);

    std::unordered_map<std::string, std::string> vertexShaders;
    std::unordered_map<std::string, std::string> fragmentShaders;

    for (const auto& file : shaderFiles) {
        std::string extension = file.substr(file.find_last_of(".") + 1);
        std::filesystem::path p(file);
        std::string name = p.stem().string();

        if (extension == "vert") vertexShaders[name] = ReadFile(file.c_str());
        else if (extension == "frag") fragmentShaders[name] = ReadFile(file.c_str());
    }

    for (const auto& vertPair : vertexShaders) {
        const std::string& name = vertPair.first;
        if (fragmentShaders.count(name)) {
            GLuint v = CompileShader(GL_VERTEX_SHADER, vertexShaders[name]);
            GLuint f = CompileShader(GL_FRAGMENT_SHADER, fragmentShaders[name]);
            shaderPrograms[name] = CreateProgram(v, f);
            glDeleteShader(v); glDeleteShader(f);
        }
    }
}

void Renderer::ProcessInput(GLFWwindow* window, float deltaTime) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) camera->ProcessKeyboard(Camera_Movement::FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) camera->ProcessKeyboard(Camera_Movement::BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) camera->ProcessKeyboard(Camera_Movement::LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) camera->ProcessKeyboard(Camera_Movement::RIGHT, deltaTime);

    static bool escPressed = false;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS && !escPressed) {
        escPressed = true;
        bool hidden = camera->switchCursor();
        glfwSetInputMode(window, GLFW_CURSOR, hidden ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_RELEASE) escPressed = false;
}

void Renderer::UpdateGridData(float* vertices, int sizeInBytes) {
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeInBytes, vertices);
}

void Renderer::RecreateGrid(const Grid& grid) {
    if (gridVAO != 0) {
        glDeleteVertexArrays(1, &gridVAO);
        glDeleteBuffers(1, &gridVBO);
        glDeleteBuffers(1, &gridEBO);
    }
    SetupGrid(grid);
}