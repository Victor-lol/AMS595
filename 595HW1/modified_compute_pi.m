
% initialize variable
precision_levels = [2, 3, 4, 5];
thresholds = [0.01, 0.001, 0.0001, 0.00001];
iterations_needed = zeros(size(precision_levels));

for p = 1:length(precision_levels)
    sigfigs = precision_levels(p);
    tolerance = thresholds(p);
    
    iter = 0;
    count = 0;
    pi_est_old = 0;
    convergence = false;
    
    tic;
    while ~convergence && iter < 1e7
        x = rand();
        y = rand();
        
        
        if x^2 + y^2 <= 1
            count = count + 1;
        end
        
        iter = iter + 1;

        % compute area
        pi_est_curr = 4 * count / iter;
        
        % avoid noise
        if iter > 100
            if abs(pi_est_curr - pi_est_old) < tolerance
                convergence = true;
            end
        end
        pi_est_old = pi_est_curr;
    end
    
    % measure time 
    elapsed = toc;

    % measure iterations
    iterations_needed(p) = iter;

    fprintf("  Precision: %d significant figures\n", sigfigs);
    fprintf("  Estimated pi: %.6f\n", pi_est_curr);
    fprintf("  Iterations required: %d\n", iter);
    fprintf("  Execution time: %.4f seconds\n\n", elapsed);
end


