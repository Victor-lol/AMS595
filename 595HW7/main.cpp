#include <iostream>
#include "number_utils.h"
#include "vector_utils.h"
#include "fib.h"
#include "prime_utils.h"
#include "pascal.h"

// Simple tests from the handout
void test_isprime() {
    std::cout << "isprime(2) = " << isprime(2) << "\n";
    std::cout << "isprime(10) = " << isprime(10) << "\n";
    std::cout << "isprime(17) = " << isprime(17) << "\n";
}

void test_factorize() {
    print_vector(factorize(2));
    print_vector(factorize(72));
    print_vector(factorize(196));
}

void test_prime_factorize() {
    print_vector(prime_factorize(2));
    print_vector(prime_factorize(72));
    print_vector(prime_factorize(196));
}

int main() {
    // Uncomment the lines you want to run.
    // report_number_sign();  // requires user input
    print_fib_upto_four_million();
    test_isprime();
    test_factorize();
    test_prime_factorize();
    print_pascals_triangle(6);
    return 0;
}
