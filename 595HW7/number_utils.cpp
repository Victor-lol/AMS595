#include <iostream>
#include "number_utils.h"

void report_number_sign() {
    int n;
    std::cout << "Enter a number: ";
    std::cin >> n;
    switch (n) {
        case -1:
            std::cout << "negative one\n";
            break;
        case 0:
            std::cout << "zero\n";
            break;
        case 1:
            std::cout << "positive one\n";
            break;
        default:
            std::cout << "other value\n";
            break;
    }
}
