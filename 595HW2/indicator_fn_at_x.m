function fn = indicator_fn_at_x(x)
% INDICATOR_FN_AT_X  Creates an indicator function along a vertical line
% INPUT:  x (real number, the x-coordinate)
% OUTPUT: fn (function handle that takes y and returns +1 or -1)
%
% Returns fn(y) where:
%   fn(y) = +1 if c = x + i*y escapes (outside Mandelbrot set)
%   fn(y) = -1 if c = x + i*y does not escape (inside Mandelbrot set)
%
% This function is used with bisection to find the boundary of the set.

    fn = @(y) ((fractal(x + 1i*y) > 0) * 2 - 1);
end
