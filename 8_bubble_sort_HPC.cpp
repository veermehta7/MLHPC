#include <iostream>
#include <vector>
#include <omp.h>

void bubbleSortParallel(std::vector<int>& arr) {
    int n = arr.size();
    bool sorted;

    #pragma omp parallel shared(arr, sorted)
    {
        do {
            sorted = true;

            #pragma omp for
            for (int i = 0; i < n - 1; ++i) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }

            #pragma omp barrier
        } while (!sorted);
    }
}

int main() {
    std::vector<int> arr = {5, 2, 8, 12, 1, 6};

    std::cout << "Original Array: ";
    for (const auto& num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    bubbleSortParallel(arr);

    std::cout << "Sorted Array: ";
    for (const auto& num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
