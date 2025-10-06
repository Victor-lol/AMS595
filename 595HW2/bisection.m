function m = bisection(fn_f, s, e)
% BISECTION  Find y where the sign of fn_f(y) changes on [s,e].
% INPUTS: fn_f (function handle y-> indicator value), s (lower y), e (upper y)
% OUTPUT: m (midpoint y where boundary lies to tolerance)
%
% Requirements:
% - Assume fn_f(s) < 0 (inside), fn_f(e) > 0 (outside).
% - Terminate when |e - s| < tol or maxIters reached.
% - Return midpoint as boundary estimate.

    tol = 1e-6;
    maxIters = 60;
    fs = fn_f(s);
    fe = fn_f(e);
    if fs * fe >= 0
        error('bisection: interval does not bracket a sign change.');
    end
    for k = 1:maxIters
        m = 0.5*(s+e);
        fm = fn_f(m);

        % Standard bisection: narrow the interval based on sign of fm
        if fm * fs < 0
            % Sign change is in [s, m], so update upper bound
            e = m;
            fe = fm;
        else
            % Sign change is in [m, e], so update lower bound
            s = m;
            fs = fm;
        end

        % Check convergence tolerance
        if abs(e - s) < tol
            break;
        end
    end
end