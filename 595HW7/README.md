# AMS 595 / DCS 525 C/C++ Project 1

This repository contains the C++17 solutions split into multiple `.h` / `.cpp` files.

## File layout
- `main.cpp`: Entry point with simple test calls (commented out by default).
- `number_utils.*`: `report_number_sign` — translation of the MATLAB `switch` task.
- `vector_utils.*`: `print_vector` — prints a `std::vector<int>`.
- `fib.*`: `print_fib_upto_four_million` — while-loop Fibonacci up to 4,000,000.
- `prime_utils.*`: `isprime`, `factorize`, `prime_factorize`.
- `pascal.*`: `print_pascals_triangle` — prints the first `n` rows of Pascal’s Triangle.

## Build
Ensure g++ with C++17 is available, then run in the repo root:
```bash
g++ -std=c++17 main.cpp number_utils.cpp vector_utils.cpp fib.cpp prime_utils.cpp pascal.cpp -o hw7
```

## Run
Uncomment the calls you want in `main()` (see below), then execute:
```bash
./hw7
```

## Quick tests available in `main.cpp`
Uncomment any of these to see output:
- `report_number_sign();`
- `print_fib_upto_four_million();`
- `test_isprime();`
- `test_factorize();`
- `test_prime_factorize();`
- `print_pascals_triangle(6);`

Rebuild after changing `main.cpp`, then run `./hw7` to view the results.
