#include <algorithm>
#include "prime_utils.h"

bool isprime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; 1LL * i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

std::vector<int> factorize(int n) {
    std::vector<int> factors;
    for (int i = 1; 1LL * i * i <= n; ++i) {
        if (n % i == 0) {
            factors.push_back(i);
            if (i != n / i) {
                factors.push_back(n / i);
            }
        }
    }
    std::sort(factors.begin(), factors.end());
    return factors;
}

std::vector<int> prime_factorize(int n) {
    std::vector<int> primes;
    if (n <= 1) return primes;
    while (n % 2 == 0) {
        primes.push_back(2);
        n /= 2;
    }
    for (int p = 3; 1LL * p * p <= n; p += 2) {
        while (n % p == 0) {
            primes.push_back(p);
            n /= p;
        }
    }
    if (n > 1) {
        primes.push_back(n);
    }
    return primes;
}
