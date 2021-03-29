%% Task
% Collects Pure datasets for different materials (friction coeff)
% Apply noise to data
% Store estimated surface properties in matrices (with labels)

%% collect Pure data
% Load finger tip sketch
load('../simulation/quarter_circle.mat')

time_simulation = 4; % [s]
g_0 = 0; % -9.82;
dim_block = [0.7, 0.2, 0.02]; % width, depth, height [m]
friction_static = 0.5; % default 0.5
friction_dynamic = 0.3; % default 0.3
radius_finger = 0.031; % [m]
length_finger = 0.05; % [m] only for cylinder
t_sample = 0.001; % [s]
pos_block = [0 0 0]; % [m] (x,y,z)

robot_height = 0.2; % [m]
robot_width = 0.6; % [m]
size_strut = 45 * 10^-3; % [m]
revolute_y = pi/4; % [rad]
revolute_x = 0; % [rad]

climit = 5; % 'threshold for cusum'

%% Load data
%out = sim('../simulation/finger', time_simulation);
%%
torque_sensor = out.Torque;
force_sensor = out.Force;
vel = out.vel.Data;
F_friction_GT = out.F_friction;
F_normal_GT = out.F_normal;
cusum_trigger = out.cusum_trigger;

force_sensor = force_sensor(logical(cusum_trigger), :);
torque_sensor = torque_sensor(logical(cusum_trigger), :);
F_friction_GT = F_friction_GT(logical(cusum_trigger), :);
F_normal_GT = F_normal_GT(logical(cusum_trigger),:);
vel = vel(logical(cusum_trigger));

%%Generate data
% data source: https://engineeringlibrary.org/reference/coefficient-of-friction
mu_s = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10 ,0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61];
mu_c = [0.42, 0.57, 0.95, 0.47, 0.36, 0.53 ,0.44, 0.21, 0.29, 1.4, 0.40, 0.48, 0.32, 0.52];
label = ["hard steel on hard steel","Mild steel on mild steel", "Mild steel on lead", "Aluminum on mild steel", "Copper on mild steel","Nickel on nickel", "Brass on mild steel","Zinc on cast iron", "Copper on cast iron", "Aluminum on aluminum", "Glass on glass", "Oak on oak (parallel to grain)", "Oak on oak (perpendicular)","Leather on oak (parallel)"];
for iter = 1:length(label)
	% set up environment
	friction_static = mu_s(iter); % default 0.5
	friction_dynamic = mu_c(iter); % default 0.3
	%simulate
	out = sim('../simulation/finger', time_simulation);
	%pass data
	torque_sensor = out.Torque;
	force_sensor = out.Force;
	vel = out.vel.Data;
	F_friction_GT = out.F_friction;
	F_normal_GT = out.F_normal;
	cusum_trigger = out.cusum_trigger;
	% choose relevant data
	force_sensor = force_sensor(logical(cusum_trigger), :);
	torque_sensor = torque_sensor(logical(cusum_trigger), :);
	F_friction_GT = F_friction_GT(logical(cusum_trigger), :);
	F_normal_GT = F_normal_GT(logical(cusum_trigger),:);
	vel = vel(logical(cusum_trigger));
	% format Data frame
	data = struct('force', force_sensor, 'torque', torque_sensor, 'vel', vel, 'friction_GT', F_friction_GT, 'normal_GT',F_normal_GT, 'static_fric_coeff', friction_static, 'dynamic_fric_coeff', friction_dynamic, 'label', label(iter));
	name = strcat('rawData/'+label(iter),'.mat');

    new_name=strrep(name," ","_");

    %save data frame
	save(new_name,'data');
end
