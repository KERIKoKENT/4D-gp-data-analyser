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

#include "src/CSVLoader.h"
#include "src/FileDialog.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <nanoflann.hpp>


struct EigenMatrixAdaptor {
    const Eigen::MatrixXd& obj;
    EigenMatrixAdaptor(const Eigen::MatrixXd& obj) : obj(obj) {}

    inline size_t kdtree_get_point_count() const { return obj.rows(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return obj(idx, dim);
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, EigenMatrixAdaptor>,
    EigenMatrixAdaptor,
    3
> MyKDTree;


bool invertConfidence = false;

glm::vec3 confidenceColor(float c, bool invert);

void updateSurface(
    std::vector<float>& vData,
    const Grid& grid,
    GP3D& gp,
    std::vector<float>& confidence,
    float sliceW,
    float centerX,
    float centerY
) {
    confidence.clear();

    int nodes = grid.getNodesSide();

    float step = grid.getStep();
    float halfSize = grid.getSize() / 2.0f;

    vData.assign(grid.getNodesCount() * 7, 0.0f);

    try {
        for (int i = 0; i < nodes; ++i) {
            for (int j = 0; j < nodes; ++j) {

                float x = i * step - halfSize + centerX;
                float y = j * step - halfSize + centerY;

                Eigen::Vector3d p;
                p << x, y, (double)sliceW;

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
    catch (const std::exception& e) {
        std::cerr << "Error during surface update: " << e.what() << std::endl;
	}
}

int main() {

    struct AppState
    {
        bool datasetLoaded = false;

        bool showStartupPopup = true;
        bool showLoadPopup = false;

        std::string path = "";

        int delimiter = ',';

        std::vector<std::string> headers;
        std::vector<std::vector<double>> data;

        int xCol = 0;
        int yCol = 1;
        int wCol = 2;
        int tCol = 3;

        float gridCenterX = 0.0f;
        float gridCenterY = 0.0f;
        float gridWidth = 20.0f;

        int kNeighbors = 1500;
        bool useLocalTraining = false;
    };

    AppState state;

    bool surfaceDirty = false;

	float currentW = 0.0;

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
    param.max_iterations = 50;

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
        double w = dist(rng);

        double z = sqrt(abs(100 - x * x - yy * yy - w * w));

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
        0.5,
        0.5,
        0.5,
        0.8,
        -1.5;


    double fx;

    try {
        solver.minimize(gp, params, fx);

        gp.fit(params);
    }
    catch (const std::exception& e) {
        std::cerr << "\n!!! FATAL ERROR !!!" << std::endl;
        std::cerr << "Message: " << e.what() << std::endl;
        std::cerr << "Params at the moment of crash: " << params.transpose() << std::endl;

        
        return -1;
	}

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

    updateSurface(vertexData, myGrid, gp, confidence, currentW, 0.0f, 0.0f);


    float defaultParams[3];

    for(unsigned int i = 0; i < 3; i++) {
        defaultParams[i] = params[i];
    }

    auto im_softplus = [](double x) {
        return std::log1p(std::exp(-std::abs(x))) + std::max(x, 0.0);
    };

    auto im_inv_softplus = [](double y)
    {
        return std::log(std::exp(y) - 1.0);
    };

    std::unique_ptr<EigenMatrixAdaptor> kdtreeAdaptor = nullptr;
    std::unique_ptr<MyKDTree> kdTree = nullptr;

    kdtreeAdaptor = std::make_unique<EigenMatrixAdaptor>(X);
    kdTree = std::make_unique<MyKDTree>(3, *kdtreeAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();

    while (!renderer.ShouldClose()) {


        renderer.PollEvents();

        renderer.Clear();

        ImGui_ImplOpenGL3_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();


        auto trainLocalModel = [&](float queryX, float queryY, float queryW) {
            if (!kdTree || X.rows() == 0) return;

            int k = std::min(state.kNeighbors, (int)X.rows());

            std::vector<size_t> ret_indexes(k);
            std::vector<double> out_dists_sq(k);

            double query_pt[3] = { (double)queryX, (double)queryY, (double)queryW };

            nanoflann::KNNResultSet<double> resultSet(k);
            resultSet.init(&ret_indexes[0], &out_dists_sq[0]);
            kdTree->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters());

            Eigen::MatrixXd localX(k, 3);
            Eigen::VectorXd localY(k);

            for (int i = 0; i < k; ++i) {
                size_t idx = ret_indexes[i];
                localX.row(i) = X.row(idx);
                localY(i) = y(idx);
            }

            gp = GP3D(localX, localY, cfg);
            double localFx;

            LBFGSpp::LBFGSParam<double> localParam;
            localParam.max_iterations = 20;
            LBFGSpp::LBFGSSolver<double> localSolver(localParam);

            try {
                localSolver.minimize(gp, params, localFx);
                gp.fit(params);
            }
            catch (...) {
                gp.fit(params);
            }
        };


        static bool firstFrame = true;

        if (firstFrame) {
            state.showStartupPopup = true;
            firstFrame = false;
        }

        if (state.showStartupPopup) {
            ImGui::OpenPopup("Startup");
        }

        if (state.showLoadPopup) {
            ImGui::OpenPopup("Load Data");
            state.showLoadPopup = false;
        }

        if (ImGui::BeginPopupModal("Startup", &state.showStartupPopup, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Load dataset to continue or explore the app");
            ImGui::Separator();

            if (ImGui::Button("Open CSV File", ImVec2(220, 0))) {
                state.showLoadPopup = true;
                state.showStartupPopup = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::Spacing();

            if (ImGui::Button("View Synthetic Test (Skip)", ImVec2(220, 0))) {
                state.showStartupPopup = false;
                ImGui::CloseCurrentPopup();
            }

            if (!state.showStartupPopup) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        if (ImGui::BeginPopupModal("Load Data", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("File");

            if (ImGui::Button("Open File")) {
                state.path = OpenFileDialog();
            }

            ImGui::Text("%s", state.path.c_str());
            ImGui::Separator();

            ImGui::Text("Delimiter");
            ImGui::RadioButton("Comma (,)", &state.delimiter, ',');
            ImGui::RadioButton("Semicolon (;)", &state.delimiter, ';');
            ImGui::RadioButton("Tab", &state.delimiter, '\t');

            ImGui::Separator();

            if (ImGui::Button("Load")) {
                state.datasetLoaded = CSVLoader::Load(state.path, (char)state.delimiter, state.headers, state.data);

                if (state.datasetLoaded) {
                    CSVLoader::BuildDataset(state.data, state.xCol, state.yCol, state.wCol, state.tCol, X, y);

                    double minX = X.col(0).minCoeff();
                    double maxX = X.col(0).maxCoeff();
                    double minY = X.col(1).minCoeff();
                    double maxY = X.col(1).maxCoeff();

                    state.gridCenterX = static_cast<float>((minX + maxX) / 2.0);
                    state.gridCenterY = static_cast<float>((minY + maxY) / 2.0);

                    float deltaX = static_cast<float>(maxX - minX);
                    float deltaY = static_cast<float>(maxY - minY);
                    state.gridWidth = std::max(deltaX, deltaY) * 1.1f;
                    if (state.gridWidth < 0.001f) state.gridWidth = 1.0f;

                    myGrid = Grid(state.gridWidth, 50);
                    renderer.SetupGrid(myGrid);

                    kdtreeAdaptor = std::make_unique<EigenMatrixAdaptor>(X);
                    kdTree = std::make_unique<MyKDTree>(3, *kdtreeAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    kdTree->buildIndex();

                    gp = GP3D(X, y, cfg);
                    solver.minimize(gp, params, fx);
                    gp.fit(params);

                    surfaceDirty = true;

                    ImGui::CloseCurrentPopup();
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        if (surfaceDirty) {
            if (state.useLocalTraining && kdTree) {
                trainLocalModel(state.gridCenterX, state.gridCenterY, currentW);
            }

            updateSurface(vertexData, myGrid, gp, confidence, currentW, state.gridCenterX, state.gridCenterY);
            surfaceDirty = false;
        }

        renderer.UpdateGridData(vertexData.data(), vertexData.size() * sizeof(float));

        renderer.DrawGrid(myGrid);


        ImGui::Begin("Settings");

        ImGui::Text("Gaussian Process");

        if (ImGui::Checkbox("Use Local KD-Tree Training", &state.useLocalTraining)) {
            surfaceDirty = true;
        }

        if (state.useLocalTraining) {

            int maxPoints = std::max((int)X.rows(), 10);
            if (state.kNeighbors > maxPoints) state.kNeighbors = maxPoints / 2;

            ImGui::SliderInt("Local Points (K)", &state.kNeighbors, 10, maxPoints);
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Training is extremely fast now!");
        }

        ImGui::Separator();


        for (int i = 0; i < 3; ++i)
        {
            ImGui::PushID(i);

            std::string label = "Lengthscale " + std::to_string(i);

            float visibleValue = im_softplus(params[i]);

            if (ImGui::SliderFloat(label.c_str(), &visibleValue, 0.01f, 20.0f))
            {
                params[i] = im_inv_softplus(visibleValue);
            }

            ImGui::SameLine();

            if (ImGui::Button("Reset"))
            {
                params[i] = defaultParams[i];
            }

            ImGui::PopID();
        }

        ImGui::Text(
            "Sigma_f: %.3f",
            im_softplus(params[3])
        );

        ImGui::Text(
            "Sigma_n: %.3f",
            im_softplus(params[4])
        );

        float wMin = -10.0f;
        float wMax = 10.0f;

        if (state.datasetLoaded && !X.col(2).isZero())
        {
            wMin = (float)X.col(2).minCoeff();
            wMax = (float)X.col(2).maxCoeff();
        }

        ImGui::SliderFloat("W-Slice ", &currentW, wMin, wMax);

        if (ImGui::Checkbox("Invert confidence", &invertConfidence)) {
            surfaceDirty = true;
        }

        if (ImGui::Button("Rebuild Surface")) {

            if (state.useLocalTraining) {
                trainLocalModel(state.gridCenterX, state.gridCenterY, currentW);
            }
            else {
                gp.fit(params);
            }

            if (X.rows() > 0) {

                double minX = X.col(0).minCoeff(); double maxX = X.col(0).maxCoeff();
                double minY = X.col(1).minCoeff(); double maxY = X.col(1).maxCoeff();

                state.gridCenterX = static_cast<float>((minX + maxX) / 2.0);
                state.gridCenterY = static_cast<float>((minY + maxY) / 2.0);

                float deltaX = static_cast<float>(maxX - minX); 
                float deltaY = static_cast<float>(maxY - minY);

                state.gridWidth = std::max(deltaX, deltaY) * 1.1f;
                if (state.gridWidth < 0.001f) state.gridWidth = 1.0f;
                myGrid = Grid(state.gridWidth, 100);
                renderer.SetupGrid(myGrid);
            }

            updateSurface(vertexData, myGrid, gp, confidence, currentW, state.gridCenterX, state.gridCenterY);

        }

        if (ImGui::Button("Load Data"))
        {
            state.showLoadPopup = true;
            ImGui::OpenPopup("Load Data");
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