%First version of the simulation
%% Load parameters
clear;
clc;
close all;

% Load finger tip sketch
load('quarter_circle.mat')

time_simulation = 7; % [s]
g = 0;
dim_block = [0.7, 0.2, 0.02]; % width, depth, height [m]
friction_dynamic = 0.3; % default 0.3
radius_finger = 0.031; % [m]
length_finger = 0.05; % [m] only for cylinder
t_sample = 0.001; % [s]
pos_block = [0 0 0]; % [m] (x,y,z)

robot_height = 0.2; % [m]
robot_width = 0.6; % [m]
size_strut = 45 * 10^-3; % [m]
revolute_y = 0;
revolute_x = 0;


%% Load data
out = sim('finger', time_simulation);

torque_sensor = out.Torque; %out.yout{1}.Values.Data;
force_sensor = out.Force; %out.yout{2}.Values.Data;

%% Run Cusum on x-force
[iupper, ilower] = cusum(force_sensor(:,1));

plot(force_sensor(:,1))
hold on
plot([ilower, iupper],[0, 0],'*')
hold off

%% Run Cusum on z-force
[iupper, ilower] = cusum(force_sensor(:,3));

plot(force_sensor(:,3))
hold on
plot([ilower, iupper],[0, 0],'*')
hold off




