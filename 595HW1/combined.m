

%%%%%%%%%%%%%%%%%%%%%%% Task1 %%%%%%%%%%%%%%%%%%%%%%%

% initalize variables
n_steps = 1e4;
errors = zeros(1, n_steps);
pi_ests = zeros(1, n_steps);
times = zeros(1, n_steps);


for k = 1:n_steps
    tic 
    count = 0;

    for i = 1:k 
        x = rand();
        y = rand();
        
        % count number of points in circle
        if x^2 + y^2 <= 1
            count = count + 1;
        end
    end

    % compute area
    pi_ests(k) = 4 * count / k;

    % compute deviation
    errors(k) = abs(pi_ests(k) - pi);

    % measure time 
    times(k) = toc;  

end

% visualization
figure;
plot(1:n_steps, pi_ests, 'b-', 'LineWidth', 1.5);
hold on;
yline(pi, 'r--', 'LineWidth', 1.5);
xlabel('Number of Points');
ylabel('Estimated \pi');
title('Monte Carlo Estimation of \pi');
legend('Estimated \pi', 'True \pi');
grid on;
saveas(gcf,'Q1_Estimated.png');

figure;
plot(1:n_steps, errors, 'm-', 'LineWidth', 1.5);
xlabel('Number of Points');
ylabel('Absolute Error/Deviation');
title('Deviation of Estimated \pi from True Value');
grid on;
saveas(gcf,'Q1_deviation.png');

figure;
scatter(times, errors, 80, 'filled');
set(gca,'XScale','log','YScale','log');
xlabel('Execution Time (s)');
ylabel('Absolute Error');
title('Precision vs Computational Cost');
grid on;

%%%%%%%%%%%%%%%%%%%%%%% Task2 %%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%% Task3 %%%%%%%%%%%%%%%%%%%%%%%
function pi_est = compute_pi(precision_level)
    
    % compute floating point position
    tolerance = 0.1 * 10^(1 - precision_level);
    
    count = 0;
    iter = 0;
    pi_est_old = 0;
    convergence = false;
    max_iter = 1e7;

    % --- Prepare plot ---
    figure; hold on;

    % Draw unit square
    rectangle('Position',[0 0 1 1],'EdgeColor','k','LineWidth',1.5);  
    
    % Draw quarter circle arc
    theta = linspace(0, pi/2, 500);
    plot(cos(theta), sin(theta), 'k-', 'LineWidth', 1.5);
    
    % Formatting
    axis equal;
    axis([0 1 0 1]);
    title('Monte Carlo Estimation of \pi');
    xlabel('x'); ylabel('y');
    grid on;

    while ~convergence && iter < max_iter
        x = rand();
        y = rand();
        

        % plot points in the graph
        if x^2 + y^2 < 1
            count = count + 1;
            plot(x, y, 'b.', 'MarkerSize', 8);
        else 
            plot(x, y, 'r.', 'MarkerSize', 8);
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
        
        % draw graph every 200 iterations
        if mod(iter, 10000) == 0
            drawnow;
        end
    end
    
    % visualization
    pi_str = num2str(pi_est_curr, precision_level);
    fprintf('Final estimate of pi (to %d significant figures): %s\n', ...
             precision_level, pi_str);

    text(0.5, -0.05, ['\pi \approx ', pi_str], ...
         'Units', 'normalized', 'FontSize', 12, 'Color', 'm', ...
         'HorizontalAlignment', 'center');
    pi_est = pi_est_curr;
end 