%% Simple test script to verify individual functions work correctly
% This tests the fractal, bisection, and poly_len functions

clear; clc;

fprintf('Testing fractal function...\n');
% Test 1: Point inside Mandelbrot set (should return 0 or high value)
c1 = 0 + 0i;  % Origin is in the set
it1 = fractal(c1);
fprintf('  fractal(0+0i) = %d (expect 0, meaning no escape)\n', it1);

% Test 2: Point outside Mandelbrot set (should escape quickly)
c2 = 2 + 2i;
it2 = fractal(c2);
fprintf('  fractal(2+2i) = %d (expect small value, escapes fast)\n', it2);

% Test 3: Point on boundary (moderate iterations)
c3 = -0.5 + 0.5i;
it3 = fractal(c3);
fprintf('  fractal(-0.5+0.5i) = %d\n', it3);

fprintf('\nTesting bisection function...\n');
% Create a simple indicator function for x = 0
fn_test = indicator_fn_at_x(0);
% Find boundary at x=0 (should be around y=1.0 for upper boundary)
y_boundary = bisection(fn_test, 0, 1.5);
fprintf('  Boundary at x=0: y = %.6f (expected ~1.0)\n', y_boundary);

fprintf('\nTesting poly_len function...\n');
% Test with a simple polynomial: y = x (straight line from 0 to 1)
% Arc length of y=x from 0 to 1 is sqrt(2) ≈ 1.414214
p_line = [1, 0];  % y = 1*x + 0
L_line = poly_len(p_line, 0, 1);
fprintf('  Length of y=x from 0 to 1: %.6f (expected ~1.414214)\n', L_line);

% Test with horizontal line: y = 0
% Arc length should be exactly 1.0
p_flat = [0];  % y = 0
L_flat = poly_len(p_flat, 0, 1);
fprintf('  Length of y=0 from 0 to 1: %.6f (expected 1.0)\n', L_flat);

fprintf('\nAll basic tests completed!\n');
