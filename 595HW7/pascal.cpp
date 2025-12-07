#include <vector>
#include "pascal.h"
#include "vector_utils.h"

void print_pascals_triangle(int n) {
    std::vector<long long> row = {1};
    for (int r = 0; r < n; ++r) {
        // print current row (cast to int for the required print function)
        print_vector(std::vector<int>(row.begin(), row.end()));
        std::vector<long long> next(row.size() + 1, 1);
        for (size_t i = 1; i < row.size(); ++i) {
            next[i] = row[i - 1] + row[i];
        }
        row.swap(next);
    }
}
