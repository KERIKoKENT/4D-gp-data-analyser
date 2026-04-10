#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h> // Добавлено для определения GLFWwindow

enum class Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);
    glm::mat4 GetViewMatrix();
    void ProcessKeyboard(Camera_Movement direction, float deltaTime);
    void ProcessCursor(GLFWwindow* window);
    void ProcessMouseMovement(float xoffset, float yoffset);
    glm::vec3 GetPosition() const;
    float GetYaw() const { return Yaw; }
    float GetPitch() const { return Pitch; }

    bool _isCursorHidden() { return isCursorHidden; }
    bool switchCursor() { isCursorHidden = !isCursorHidden; return isCursorHidden; }

private:
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    float Yaw;
    float Pitch;
    float MovementSpeed;
    float MouseSensitivity;
    bool isCursorHidden;

    void updateCameraVectors();
};

#endif