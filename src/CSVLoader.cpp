#include "CSVLoader.h"

#include <fstream>
#include <sstream>

bool CSVLoader::Load(
    const std::string& path,
    char delimiter,
    std::vector<std::string>& headers,
    std::vector<std::vector<double>>& data)
{
    std::ifstream file(path);
    if (!file.is_open()) return false;

    headers.clear();
    data.clear();

    std::string line;

    if (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter))
            headers.push_back(cell);
    }

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        std::vector<double> row;

        while (std::getline(ss, cell, delimiter))
            row.push_back(std::stod(cell));

        if (!row.empty())
            data.push_back(row);
    }

    return true;
}

void CSVLoader::BuildDataset(
    const std::vector<std::vector<double>>& data,
    int xCol, int yCol, int wCol, int tCol,
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y)
{
    int N = (int)data.size();

    X.resize(N, 3);
    y.resize(N);

    for (int i = 0; i < N; i++)
    {
        const auto& r = data[i];

        X(i, 0) = r[xCol];
        X(i, 1) = r[yCol];
        X(i, 2) = r[wCol];

        y(i) = r[tCol];
    }
}