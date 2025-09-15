
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
        if mod(iter, 200) == 0
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