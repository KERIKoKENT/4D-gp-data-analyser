#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "LBFGS.h"
#include "src/Renderer.h"
#include "src/Grid.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

void updateSurface(std::vector<float>& vData, const Grid& grid) {
    int nodes = grid.getNodesSide();
    float step = grid.getStep();
    float halfSize = grid.getSize() / 2.0f;

    vData.assign(grid.getNodesCount() * 6, 0.0f);

    auto f = [](float x, float y) {
        return (float)(sin(x) * cos(y));
    };

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
            float x = i * step - halfSize;
            float y = j * step - halfSize;
            float z = f(x, y);

            int idx = (i * nodes + j) * 6;
            vData[idx] = x;
            vData[idx + 1] = y;
            vData[idx + 2] = z;
        }
    }

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
            int idx = (i * nodes + j) * 6;

            float z_left = (i > 0) ? vData[((i - 1) * nodes + j) * 6 + 2] : vData[idx + 2];
            float z_right = (i < nodes - 1) ? vData[((i + 1) * nodes + j) * 6 + 2] : vData[idx + 2];
            float z_down = (j > 0) ? vData[(i * nodes + (j - 1)) * 6 + 2] : vData[idx + 2];
            float z_up = (j < nodes - 1) ? vData[(i * nodes + (j + 1)) * 6 + 2] : vData[idx + 2];

            float dzdx = (z_right - z_left) / (2.0f * step);
            float dzdy = (z_up - z_down) / (2.0f * step);

            glm::vec3 n = glm::normalize(glm::vec3(-dzdx, -dzdy, 1.0f));

            vData[idx + 3] = n.x;
            vData[idx + 4] = n.y;
            vData[idx + 5] = n.z;
        }
    }
}

int main() {
    Renderer renderer(1600, 900, "3D Plotter");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(renderer.GetWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 330");

    LBFGSpp::LBFGSParam<double> param;
    LBFGSpp::LBFGSSolver<double> solver(param);
    Eigen::VectorXd x_min(2);
    x_min << 0.0, 0.0;
    auto func = [](const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
        double X = x[0], Y = x[1];
        grad[0] = 4 * X * (X * X + Y - 11) + 2 * (X + Y * Y - 7);
        grad[1] = 2 * (X * X + Y - 11) + 4 * Y * (X + Y * Y - 7);
        return (X * X + Y - 11) * (X * X + Y - 11) + (X + Y * Y - 7) * (X + Y * Y - 7);
        };

    Grid myGrid(20.0f, 50);
    renderer.SetupGrid(myGrid);

    std::vector<float> vertexData;
    updateSurface(vertexData, myGrid);

    //Тестовый вывод всех вершин
    std::cout << "--- START VERTEX DATA TEST ---" << std::endl;
    for (size_t i = 0; i < vertexData.size(); i += 6) {
        std::cout << "Node " << i / 6
            << " | Pos: [" << vertexData[i] << ", " << vertexData[i + 1] << ", " << vertexData[i + 2] << "]"
            << " | Norm: [" << vertexData[i + 3] << ", " << vertexData[i + 4] << ", " << vertexData[i + 5] << "]"
            << std::endl;
    }
    std::cout << "--- END VERTEX DATA TEST (Total: " << vertexData.size() / 6 << " nodes) ---" << std::endl;


    while (!renderer.ShouldClose()) {
        renderer.PollEvents();
        renderer.Clear();

        renderer.UpdateGridData(vertexData.data(), vertexData.size() * sizeof(float));
        renderer.DrawGrid(myGrid);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Settings");
        if (ImGui::Button("Solve L-BFGS")) {
            double fx;
            solver.minimize(func, x_min, fx);
            std::cout << "Min: " << x_min.transpose() << " Value: " << fx << std::endl;
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        renderer.SwapBuffers();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return 0;
}