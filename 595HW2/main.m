%% AMS 595 / DCS 525 Project 2 - Main script
% How Long Is the Coast of Britain? (Mandelbrot boundary & length)
% NOTE: Fill in the TODOs in your functions first.
% This script: samples x, finds boundary y(x), fits poly, integrates length.

clear; clc; close all;

%% 0) Quick Mandelbrot heatmap (optional but helpful for choosing bounds)
% Create a coarse grid to visualize escape iterations.
% This helps pick reasonable y-upper bounds per x.
X = linspace(-2.0, 1.0, 400);
Y = linspace(-1.5, 1.5, 300);
IT = zeros(numel(Y), numel(X));
for iy = 1:numel(Y)
    for ix = 1:numel(X)
        IT(iy,ix) = fractal(X(ix) + 1i*Y(iy));
    end
end
figure; imshow(IT, [], 'XData', [X(1), X(end)], 'YData', [Y(1), Y(end)]);
axis on; xlabel('Re(c)'); ylabel('Im(c)'); title('Mandelbrot Iterations');

%% 1) Sample x's and find boundary y(x) via bisection
% Sampling: at least 1e3 points in [-2,1].
Nx = 1200;
xs = linspace(-2, 1, Nx);
ys = nan(size(xs));

% For bisection, we need y-lower (inside) and y-upper (outside).
% Inside near y=0 usually works for the main cardioid; choose a line spanning set vertically.
for i = 1:Nx
    x = xs(i);
    fn = indicator_fn_at_x(x);

    % Heuristic vertical bracket: start inside at y=0; march upward to find outside.
    yl = 0.0;             % expect inside near center
    yu = 1.5;             % top of our plotted window; adjust if needed

    % Ensure inside/outside; if yl is not inside, nudge downward.
    if fn(yl) > 0
        yl = -1.5;  % try bottom
        if fn(yl) > 0
            % Couldn't find inside at this x; skip (flat tails / outside entire column)
            continue;
        end
    end
    % Ensure yu is outside; if not, raise it a bit (or skip this x if always inside).
    if fn(yu) < 0
        % still inside at top; skip (this x may be entirely interior)
        continue;
    end

    ys(i) = bisection(fn, yl, yu);
end

% Keep only finite boundary points
mask = isfinite(ys);
x_b = xs(mask);
y_b = ys(mask);

figure; plot(x_b, y_b, '.', 'MarkerSize', 4);
xlabel('x'); ylabel('y'); title('Estimated boundary points (upper branch)');
grid on;

%% 2) Trim to the real boundary span before fitting
% The boundary goes flat far left/right; you must discard those regions
% to avoid wrecking the poly fit. Manually tune [xmin, xmax] by
% looking at the scatter above.
% Good starting values based on Mandelbrot geometry:
xmin = -1.4;   % <- tune after plotting to exclude flat left region
xmax =  0.4;   % <- tune after plotting to exclude flat right region
sel = (x_b >= xmin) & (x_b <= xmax);
x_fit = x_b(sel);
y_fit = y_b(sel);

%% 3) Fit a degree-15 polynomial
order = 15;
p = polyfit(x_fit, y_fit, order);

% Visual check
xx = linspace(min(x_fit), max(x_fit), 2000);
yy = polyval(p, xx);
figure; plot(x_fit, y_fit, '.', 'MarkerSize', 4); hold on;
plot(xx, yy, 'LineWidth', 1.5);
xlabel('x'); ylabel('y'); title(sprintf('Order-%d polynomial fit', order));
legend('Boundary samples','Poly fit'); grid on;

%% 4) Arc length on the same trimmed interval
L = poly_len(p, min(x_fit), max(x_fit));
fprintf('Approximate boundary length on [%g,%g]:  L = %.6f\n', ...
        min(x_fit), max(x_fit), L);
