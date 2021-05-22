%% Load data
clear
clc
close all
centroids_pi4_all_int = readmatrix('_pi4_AllCenters_int.csv');
centroids_pi4_Alu_int = readmatrix('A_pi4_Centers_int.csv');
centroids_pi4_Pap_int = readmatrix('C_pi4_Centers_int.csv');
centroids_pi4_Tree_int = readmatrix('T_pi4_Centers_int.csv');

centroids_pi_all_int = readmatrix('_pi_AllCenters_int.csv');
centroids_pi_Alu_int = readmatrix('A_pi_Centers_int.csv');
centroids_pi_Pap_int = readmatrix('C_pi_Centers_int.csv');
centroids_pi_Tree_int = readmatrix('T_pi_Centers_int.csv');

centroids_pi4_all_ext = readmatrix('_pi4_AllCenters_ext.csv');
centroids_pi4_Alu_ext = readmatrix('A_pi4_Centers_ext.csv');
centroids_pi4_Pap_ext = readmatrix('C_pi4_Centers_ext.csv');
centroids_pi4_Tree_ext = readmatrix('T_pi4_Centers_ext.csv');

centroids_pi_all_ext = readmatrix('_pi_AllCenters_ext.csv');
centroids_pi_Alu_ext = readmatrix('A_pi_Centers_ext.csv');
centroids_pi_Pap_ext = readmatrix('C_pi_Centers_ext.csv');
centroids_pi_Tree_ext = readmatrix('T_pi_Centers_ext.csv');

centroids_conv_pi4_all_int = readmatrix('_conv_pi4_AllCenters_int.csv');
centroids_conv_pi4_Alu_int = readmatrix('A_conv_pi4_Centers_int.csv');
centroids_conv_pi4_Pap_int = readmatrix('C_conv_pi4_Centers_int.csv');
centroids_conv_pi4_Tree_int = readmatrix('T_conv_pi4_Centers_int.csv');

centroids_conv_pi_all_int = readmatrix('_conv_pi_AllCenters_int.csv');
centroids_conv_pi_Alu_int = readmatrix('A_conv_pi_Centers_int.csv');
centroids_conv_pi_Pap_int = readmatrix('C_conv_pi_Centers_int.csv');
centroids_conv_pi_Tree_int = readmatrix('T_conv_pi_Centers_int.csv');

centroids_conv_pi4_all_ext = readmatrix('_conv_pi4_AllCenters_ext.csv');
centroids_conv_pi4_Alu_ext = readmatrix('A_conv_pi4_Centers_ext.csv');
centroids_conv_pi4_Pap_ext = readmatrix('C_conv_pi4_Centers_ext.csv');
centroids_conv_pi4_Tree_ext = readmatrix('T_conv_pi4_Centers_ext.csv');

centroids_conv_pi_all_ext = readmatrix('_conv_pi_AllCenters_ext.csv');
centroids_conv_pi_Alu_ext = readmatrix('A_conv_pi_Centers_ext.csv');
centroids_conv_pi_Pap_ext = readmatrix('C_conv_pi_Centers_ext.csv');
centroids_conv_pi_Tree_ext = readmatrix('T_conv_pi_Centers_ext.csv');

%% Calculate ground truth
r = 31*10^(-3);
x_GT_pi4 = 0;
y_GT_pi4 = sin(pi/4)*r;
z_GT_pi4 = cos(pi/4)*r;

x_GT_pi = 0;
y_GT_pi = 0;
z_GT_pi = r;

%% Calculate errors

x_error_conv_pi_all = centroids_conv_pi_all_ext(:,1) - x_GT_pi;
y_error_conv_pi_all = centroids_conv_pi_all_ext(:,2) - y_GT_pi;
z_error_conv_pi_all = centroids_conv_pi_all_ext(:,3) - z_GT_pi;

x_error_conv_pi4_all = abs(centroids_conv_pi4_all_ext(:,1)) - x_GT_pi4;
y_error_conv_pi4_all = abs(centroids_conv_pi4_all_ext(:,2)) - y_GT_pi4;
z_error_conv_pi4_all = abs(centroids_conv_pi4_all_ext(:,3)) - z_GT_pi4;

mean(x_error_conv_pi4_all)  
mean(y_error_conv_pi4_all)
mean(z_error_conv_pi4_all)

x_error_conv_pi_all_int = centroids_conv_pi_all_int(:,1) - x_GT_pi;
y_error_conv_pi_all_int = centroids_conv_pi_all_int(:,2) - y_GT_pi;
z_error_conv_pi_all_int = centroids_conv_pi_all_int(:,3) - z_GT_pi;

x_error_conv_pi4_all_int = centroids_conv_pi4_all_int(:,1) - x_GT_pi4;
y_error_conv_pi4_all_int = centroids_conv_pi4_all_int(:,2) - y_GT_pi4;
z_error_conv_pi4_all_int = centroids_conv_pi4_all_int(:,3) - z_GT_pi4;


%%
mean_pap_pi = mean(centroids_pi_Pap)
mean_alu_pi = mean(centroids_pi_Alu)
mean_tree_pi = mean(centroids_pi_Tree)

mean_pap_pi4 = mean(centroids_pi4_Pap)
mean_alu_pi4 = mean(centroids_pi4_Alu)
mean_tree_pi4 = mean(centroids_pi4_Tree)


%% conv_pi_all
x = 1:3;
%data_mean = [mean(x_error_conv_pi_all); mean(y_error_conv_pi_all); mean(z_error_conv_pi_all)];
%stds = [std(x_error_conv_pi_all), std(y_error_conv_pi_all), std(z_error_conv_pi_all)];



data_mean = [mean(x_error_conv_pi4_all); mean(y_error_conv_pi4_all); mean(z_error_conv_pi4_all)];
%data_mean2 = [mean(x_error_conv_pi4_all), mean(y_error_conv_pi4_all), mean(z_error_conv_pi4_all)]
stds = [std(x_error_conv_pi4_all), std(y_error_conv_pi4_all), std(z_error_conv_pi4_all)];


%data = [x_error_conv_pi_all; y_error_conv_pi_all; z_error_conv_pi_all];
bar(x, abs(data_mean))
hold on
er = errorbar(x, abs(data_mean), stds, stds)
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

xlabel('Axis')
xticklabels({'X', 'Y', 'Z'})
ylabel('Mean error [m]')
set(gca,'FontSize',18)

