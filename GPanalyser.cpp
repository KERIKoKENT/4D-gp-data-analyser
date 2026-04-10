#include <iostream>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <Eigen/Dense>
#include "LBFGS.h"

int main() {
    if (!glfwInit()) return -1;

    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Window", NULL, NULL);
    if (!window) return -1;

    glfwMakeContextCurrent(window);
    glViewport(0, 0, 1600, 900);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    LBFGSpp::LBFGSParam<double> param;  
    LBFGSpp::LBFGSSolver<double> solver(param);

    Eigen::VectorXd x(2);               
    x << 0.0, 0.0;                       

    auto func = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
        double X = x[0], Y = x[1];
        grad[0] = 4 * X * (X*X + Y - 11) + 2*(X + Y*Y - 7);
        grad[1] = 2*(X*X + Y - 11) + 4*Y*(X + Y*Y - 7);
        return (X*X + Y - 11) * (X * X + Y - 11) + (X + Y*Y - 7) * (X + Y * Y - 7);
        };

    double fx;
    int niter = solver.minimize(func, x, fx);

    std::cout << "Minimum in x: " << x.transpose() << "\n";
    std::cout << "Value: " << fx << "\n";
    std::cout << "Iters: " << niter << "\n";

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Hello");
        ImGui::Text("It works!");
        ImGui::End();

        ImGui::Render();
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}