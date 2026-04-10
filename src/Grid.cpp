#include "Grid.h"

Grid::Grid(float size, int nodesSide) : Size(size), NodesSide(nodesSide) {}

void Grid::generateIndices(unsigned int* arr) const {
    int k = 0;
    for (int i = 0; i < NodesSide - 1; ++i) {
        for (int j = 0; j < NodesSide - 1; ++j) {
            unsigned int topLeft = i * NodesSide + j;
            unsigned int topRight = i * NodesSide + j + 1;
            unsigned int bottomLeft = (i + 1) * NodesSide + j;
            unsigned int bottomRight = (i + 1) * NodesSide + j + 1;

            arr[k++] = topLeft;
            arr[k++] = bottomLeft;
            arr[k++] = topRight;

            arr[k++] = topRight;
            arr[k++] = bottomLeft;
            arr[k++] = bottomRight;
        }
    }
}
