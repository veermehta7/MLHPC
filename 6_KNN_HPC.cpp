#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

struct Point {
    double x, y;
    int label;
};

double euclideanDistance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

int knn(const std::vector<Point>& trainingSet, const Point& query, int k) {
    std::vector<double> distances(trainingSet.size());

    #pragma omp parallel for
    for (int i = 0; i < trainingSet.size(); ++i) {
        distances[i] = euclideanDistance(trainingSet[i], query);
    }

    std::vector<int> indices(trainingSet.size());
    for (int i = 0; i < trainingSet.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on distances in ascending order
    std::sort(indices.begin(), indices.end(), [&distances](int i, int j) {
        return distances[i] < distances[j];
    });

    // Count the labels of k nearest neighbors
    std::vector<int> labelCount(10);  // Assuming labels are in the range 0-9
    for (int i = 0; i < k; ++i) {
        int index = indices[i];
        int label = trainingSet[index].label;
        ++labelCount[label];
    }

    // Find the label with maximum count
    int maxCount = 0;
    int maxLabel = 0;
    for (int i = 0; i < 10; ++i) {
        if (labelCount[i] > maxCount) {
            maxCount = labelCount[i];
            maxLabel = i;
        }
    }

    return maxLabel;
}

int main() {
    // Sample training set
    std::vector<Point> trainingSet = {
        {1.0, 2.0, 0},
        {2.0, 1.0, 1},
        {3.0, 4.0, 1},
        {4.0, 3.0, 0},
        {5.0, 6.0, 1},
        {6.0, 5.0, 0}
    };

    // Sample query point
    Point query = {3.5, 2.5, 0};

    // Number of nearest neighbors to consider
    int k = 3;

    int result = knn(trainingSet, query, k);

    std::cout << "Query point belongs to class: " << result << std::endl;

    return 0;
}
