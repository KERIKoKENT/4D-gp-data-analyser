#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

class CSVLoader
{
public:
    static bool Load(
        const std::string& path,
        char delimiter,
        std::vector<std::string>& headers,
        std::vector<std::vector<double>>& data
    );

    static void BuildDataset(
        const std::vector<std::vector<double>>& data,
        int xCol, int yCol, int wCol, int tCol,
        Eigen::MatrixXd& X,
        Eigen::VectorXd& y
    );
};