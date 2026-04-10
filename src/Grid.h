#pragma once

class Grid {
public:
    Grid(float size, int nodesSide);

    int getNodesSide() const { return NodesSide; }
    float getSize() const { return Size; }
    float getStep() const { return Size / (NodesSide - 1); }

    unsigned int getNodesCount() const { return NodesSide * NodesSide; }
    int getIndicesCount() const { return (NodesSide - 1) * (NodesSide - 1) * 6; }

    void generateIndices(unsigned int* arr) const;

private:
    float Size;
    int NodesSide;
};