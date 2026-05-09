#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "LBFGS.h"

#include "src/Renderer.h"
#include "src/Grid.h"

#include "src/gp3d.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

bool invertConfidence = false;

glm::vec3 confidenceColor(float c, bool invert);

void updateSurface(
    std::vector<float>& vData,
    const Grid& grid,
    GP3D& gp,
    std::vector<float>& confidence
) {
    confidence.clear();

    int nodes = grid.getNodesSide();

    float step = grid.getStep();
    float halfSize = grid.getSize() / 2.0f;

    vData.assign(grid.getNodesCount() * 7, 0.0f);

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {

            float x = i * step - halfSize;
            float y = j * step - halfSize;

            Eigen::Vector3d p;
            p << x, y, 0.0;

            auto [mean, var] = gp.predict(p);

            float z = (float)mean;

            float conf = exp(-(float)var * 3.0f);

            if (invertConfidence)
                conf = 1.0f - conf;

            confidence.push_back(conf);

            int idx = (i * nodes + j) * 7;

            vData[idx + 0] = x;
            vData[idx + 1] = y;
            vData[idx + 2] = z;

            vData[idx + 3] = conf;
        }
    }

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {

            int idx = (i * nodes + j) * 7;

            float z_left =
                (i > 0)
                ? vData[((i - 1) * nodes + j) * 7 + 2]
                : vData[idx + 2];

            float z_right =
                (i < nodes - 1)
                ? vData[((i + 1) * nodes + j) * 7 + 2]
                : vData[idx + 2];

            float z_down =
                (j > 0)
                ? vData[(i * nodes + (j - 1)) * 7 + 2]
                : vData[idx + 2];

            float z_up =
                (j < nodes - 1)
                ? vData[(i * nodes + (j + 1)) * 7 + 2]
                : vData[idx + 2];

            float dzdx = (z_right - z_left) / (2.0f * step);
            float dzdy = (z_up - z_down) / (2.0f * step);

            glm::vec3 n =
                glm::normalize(glm::vec3(-dzdx, -dzdy, 1.0f));

            vData[idx + 4] = n.x;
            vData[idx + 5] = n.y;
            vData[idx + 6] = n.z;
        }
    }
}

int main() {

    bool surfaceDirty = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    Renderer renderer(1600, 900, "3D Plotter");

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(
        renderer.GetWindow(),
        true
    );

    ImGui_ImplOpenGL3_Init("#version 330");

    LBFGSpp::LBFGSParam<double> param;
    param.max_iterations = 100;

    LBFGSpp::LBFGSSolver<double> solver(param);

    const int N = 300;

    Eigen::MatrixXd X(N, 3);
    Eigen::VectorXd y(N);

    std::mt19937 rng(42);

    std::uniform_real_distribution<double>
        dist(-10.0, 10.0);

    std::normal_distribution<double>
        noise(0.0, 0.03);

    for (int i = 0; i < N; i++) {

        double x = dist(rng);
        double yy = dist(rng);
        double w = 0.0;

        double z =
            sin(x) * cos(yy)
            + 0.5 * tanh(x - yy)
            + 0.2 * sin(x * yy);

        X(i, 0) = x;
        X(i, 1) = yy;
        X(i, 2) = w;

        y(i) = z;
    }

    std::vector<float> confidence;

    GPConfig cfg;

    GP3D gp(X, y, cfg);

    Eigen::VectorXd params(5);

    params <<
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0;


    double fx;

    solver.minimize(gp, params, fx);

    gp.fit(params);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "X took: "
        << std::chrono::duration<double>(t1 - t0).count()
        << "s\n";
    t0 = t1;

    std::cout << "GP trained\n";

    std::cout
        << "Params: "
        << params.transpose()
        << std::endl;

    Grid myGrid(20.0f, 100);

    renderer.SetupGrid(myGrid);

    std::vector<float> vertexData;

    updateSurface(vertexData, myGrid, gp, confidence);

    while (!renderer.ShouldClose()) {

        if (surfaceDirty) {
            updateSurface(vertexData, myGrid, gp, confidence);
            surfaceDirty = false;
        }

        renderer.PollEvents();

        renderer.Clear();

        renderer.UpdateGridData(
            vertexData.data(),
            vertexData.size() * sizeof(float)
        );

        renderer.DrawGrid(myGrid);

        ImGui_ImplOpenGL3_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        ImGui::Begin("Settings");

        ImGui::Text("Gaussian Process");

        ImGui::Text(
            "Lengthscales: %.3f %.3f %.3f",
            std::exp(params[0]),
            std::exp(params[1]),
            std::exp(params[2])
        );

        ImGui::Text(
            "Sigma_f: %.3f",
            std::exp(params[3])
        );

        ImGui::Text(
            "Sigma_n: %.3f",
            std::exp(params[4])
        );

        if (ImGui::Checkbox("Invert confidence", &invertConfidence)) {
            surfaceDirty = true;
        }

        if (ImGui::Button("Rebuild Surface")) {
            updateSurface(vertexData, myGrid, gp, confidence);
        }

        ImGui::End();

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData()
        );

        renderer.SwapBuffers();
    }

    ImGui_ImplOpenGL3_Shutdown();

    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    return 0;
}