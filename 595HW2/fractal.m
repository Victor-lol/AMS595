function it = fractal(c)
% FRACTAL  Return the iteration count until |z|>2 or maxed out.
% INPUT:  c (complex scalar)
% OUTPUT: it (integer, 0..maxIter)  -- number of iterations before escape
%
% Hints:
% - Use z_{k+1} = z_k^2 + c, z_0 = 0.
% - Stop once abs(z)>2 OR you hit maxIter.
% - Return the iteration index at which it escaped; 0 if it never escaped.

    maxIter = 100;
    z = 0;
    it = 0;  % stays 0 if never escapes within maxIter

    % Iterate the formula z_{k+1} = z_k^2 + c starting from z_0 = 0
    for k = 1:maxIter
        z = z^2 + c;  % Apply the Mandelbrot iteration formula
        if abs(z) > 2.0
            it = k;  % Record the iteration count at which escape occurs
            return;  % Exit early once we've exceeded the threshold
        end
    end
    % If we reach here, point did not escape within maxIter iterations
    % it remains 0
end