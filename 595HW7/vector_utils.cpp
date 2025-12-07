#include <iostream>
#include "vector_utils.h"

void print_vector(const std::vector<int> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) {
            std::cout << " ";
        }
    }
    std::cout << "\n";
}
