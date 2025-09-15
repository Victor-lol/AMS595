


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

saveas(gcf,'Q1_precision.png');