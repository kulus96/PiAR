%% Initialize simulation
clear;
clc;
close all;

% Load finger tip sketch
load('../simulation/quarter_circle.mat')

time_simulation = 4; % [s]
g_0 = 0; % -9.82;
dim_block = [0.7, 0.2, 0.02]; % width, depth, height [m]
friction_static = 0.5; % default 0.5
friction_dynamic = 0.7; % default 0.3
radius_finger = 0.031; % [m]
length_finger = 0.05; % [m] only for cylinder
t_sample = 0.001; % [s]
pos_block = [0 0 0]; % [m] (x,y,z)

robot_height = 0.2; % [m]
robot_width = 0.6; % [m]
size_strut = 45 * 10^-3; % [m]
revolute_y = pi/4; % [rad]
revolute_x = 0; % [rad]
F_push = -5;
climit = 28; % 'threshold for cusum'


%% Run simulation
out = sim('../simulation/finger', time_simulation);

%% Load data
torque_sensor = out.Torque;
force_sensor = out.Force;
vel = out.vel.Data;
F_friction_GT = out.F_friction;
F_normal_GT = out.F_normal;
cusum_trigger = out.cusum_trigger;

% Apply cusum trigger
force_sensor = force_sensor(logical(cusum_trigger), :);
torque_sensor = torque_sensor(logical(cusum_trigger), :);
F_friction_GT = F_friction_GT(logical(cusum_trigger), :);
vel = vel(logical(cusum_trigger));
F_normal_GT = F_normal_GT(logical(cusum_trigger), :);

% Down sample data by every n_th
n_th = 4;
forces_tcp = force_sensor(1:n_th:end,:)';
moments_tcp = torque_sensor(1:n_th:end,:)';
vel_tcp = vel(1:n_th:end);
F_friction_GT = F_friction_GT(1:4:end,:);

%% Estimate contact point
% Surface parameters
R = 31 * 10^-3; % [m]
A = eye(3);
A(3,3)=1;

%p_rectified= [sin(pi/5) 0 cos(pi/5)]'; % pi/5
p_rectified= [sin(pi/4)*R 0 cos(pi/4)*R]'; % pi/4
%p_rectified= [0 0 1]'; % 0 deg

syms x y z k;
p = [x,y,z]';
r = R;
dz = 0;

% Variable for plotting
x_cord = zeros(1,length(forces_tcp));
y_cord = zeros(1,length(forces_tcp));
z_cord = zeros(1,length(forces_tcp));
x_iter_cord = zeros(1,length(forces_tcp));
y_iter_cord = zeros(1,length(forces_tcp));
z_iter_cord = zeros(1,length(forces_tcp));
k = zeros(1,length(forces_tcp));
k_iter = zeros(1,length(forces_tcp));
fm_ratio = zeros(1,length(forces_tcp));
F_normals = zeros(length(forces_tcp),1);
F_frictions = zeros(length(forces_tcp),1);
radius_check = zeros(1, length(forces_tcp));
execution_time_analytical = zeros(1, length(forces_tcp));
execution_time_numerical = zeros(1, length(forces_tcp));

% Functions for the numerical method
S = @(p)p(1)^2+p(2)^2+(p(3)-dz)^2-r^2; % Function for sphere surface
g = @(x0, xdata)[cross(xdata(1:3),x0(1:3)) + x0(4)*gradient(S(x0(1:3))) - xdata(4:6); S(x0(1:3))]; % Objective fucntion for moments and dist to surface


options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');


for i = 1:length(forces_tcp)
    f = forces_tcp(:,i);
    m = moments_tcp(:,i);
    
    % Analytical solution on a sphere
    fm_ratio(i)= f'*m;
    tic
    [c_sph, k_sph] = contactCentroidEstimation(A, R, f, m);
    execution_time_analytical(i) = toc;
    x_cord(i) = c_sph(1);
    y_cord(i) = c_sph(2);
    z_cord(i) = c_sph(3);
    k(i) = k_sph;
    
    % Numerical solution using the Levenberg-Marquardt method
    ydata = zeros(4,1);
    x0 = [[0;0;10.0]; k_sph]; % Initial guess
    xdata = [f;m];
    tic;
    [x,~,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(g,x0,xdata, ydata, [], [], options);
    execution_time_numerical(i) = toc;
    x_iter_cord(i) = x(1);
    y_iter_cord(i) = x(2);
    z_iter_cord(i) = x(3);
    k_iter(i) = x(4);
    
    % Estimate friction force
    P_c = [x(1),x(2),x(3)]';
    [F_normal, F_friction, F_friction2] = estimateFriction(P_c,f,S);
    F_normals(i) = norm(F_normal);
    F_frictions(i) = norm(F_friction);      
end
disp('== Done estimating forces ===');


%%
% Estimate friction coefficients
% x0 = [mu_s, mu_c, v0, nabla]
x0 = [0, 0.7, 0, 0]';
%F_n v F_fric

n_data = 50;
n_start = 300;
xdata = [F_normals(n_start:n_start+n_data,:), vel_tcp(n_start:n_start+n_data), F_frictions(n_start:n_start+n_data,:)];
ydata = zeros(length(xdata), 1);

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt', 'PlotFcn', 'optimplotx', 'Diagnostics','on','Display','iter-detailed', 'MaxIterations', 100);
lb = [];    % upper bound
ub = [];    % lower bound

tic;
[x_, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(@g_func, x0, xdata, ydata, lb, ub, options);
display(toc)
display(x_)

% Stribeck GUESSstimation: 0.0331
% mu_str = vel_tcp(80)*sqrt(2);

%% Plot contact point
close all;
figure(1)
subplot(4,1,1)
plot(1:length(forces_tcp), x_cord, 1:length(forces_tcp), x_iter_cord, 1:length(forces_tcp), repmat(p_rectified(1),1,length(forces_tcp)));
title('Contact point')
ylabel('X coordinate [mm]')
%ylim([-2;2])

legend('Analytical', 'Numerical', 'Ground truth', 'Location', 'eastoutside')
subplot(4,1,2)
plot(1:length(forces_tcp), y_cord, 1:length(forces_tcp), y_iter_cord,1:length(forces_tcp), repmat(p_rectified(2),1,length(forces_tcp)));
ylabel('Y coordinate [mm]')
%ylim([-2;2])

subplot(4,1,3)
plot(1:length(forces_tcp), z_cord, 1:length(forces_tcp), z_iter_cord, 1:length(forces_tcp), repmat(p_rectified(3),1,length(forces_tcp)));
ylabel('Z coordinate [mm]')
%ylim([-2;2])

subplot(4,1,4)
plot(1:length(forces_tcp), k, 'o',1:length(forces_tcp), k_iter,'.');
ylabel('K Value')
xlabel('N iteration')

%% Plot execution times

figure(4)
subplot(2,1,1)
plot(1:length(forces_tcp), execution_time_analytical)
title('Execution time of the analytical method')
ylabel('Time [s]')

subplot(2,1,2)
plot(1:length(forces_tcp), execution_time_numerical)
title('Execution time of the numerical method')
ylabel('Time [s]')

%% Plot friction
figure(5)
plot(F_frictions,'.')
hold on
plot(F_friction_GT,'o')
hold off
title('Friction force')
xlabel('Time [ms]')
ylabel('Force [N]')
legend('Estimated friction force', 'Ground truth')


figure(6)
plot(F_frictions-F_friction_GT,'r*')
title('Friction error')
xlabel('Time [ms]')
ylabel('Force [N]')




