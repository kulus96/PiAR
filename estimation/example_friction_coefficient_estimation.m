%% Load data
clear;
clc;
close all;
[forces_tcp, moments_tcp, vel_tcp, F_friction_GT, F_normal_GT, mu_static_GT, mu_dyn_GT, label] = loadRawData('rawData/Mild_steel_on_lead.mat', 8, 0);

%%
[mu_static, mu_dynamic, stribeck, nabla] = estimateFrictionCoefficients(forces_tcp, moments_tcp, vel_tcp);



