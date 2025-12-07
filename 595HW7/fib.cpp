#include <iostream>
#include "fib.h"

void print_fib_upto_four_million() {
    long long a = 1;
    long long b = 2;
    while (a <= 4'000'000) {
        std::cout << a << " ";
        long long next = a + b;
        a = b;
        b = next;
    }
    std::cout << "\n";
}
