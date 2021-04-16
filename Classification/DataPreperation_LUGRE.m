clear; clc;
addpath("./../estimation")
% load data
files = dir('../estimation/rawData/*.mat');


% Collum = [Static, dynamiv, Stribeck vel, viscosity ]
SURFACE_PROPERTIES =[]; 
GT_STATIC = [];
GT_DYNAMIC =[];
LABEL = [];

% Data variation parameters
n_samples_pr_class = 2;
noise_std =0;

for i = 1:size(files,1)
    % load data
    [forces_tcp, moments_tcp, vel_tcp, F_friction_GT, F_normal_GT, mu_static_GT, mu_dyn_GT, label] = loadRawData(strcat(files(i).folder,'/', files(i).name), 8, true);
    % resample 
    for k = 1:n_samples_pr_class
        % estimate model param
        [mu_static, mu_dynamic, stribeck, nabla] = estimateFrictionCoefficients(forces_tcp, moments_tcp, vel_tcp);
        % Order the data structure
        LuGre_surface_property = [mu_static, mu_dynamic, stribeck, nabla];
        % Save to data matrix
        SURFACE_PROPERTIES = [SURFACE_PROPERTIES; LuGre_surface_property];
        GT_STATIC = [GT_STATIC; mu_static_GT];
        GT_DYNAMIC = [GT_DYNAMIC; mu_dyn_GT];
        LABEL = [LABEL; label];
    end
end


