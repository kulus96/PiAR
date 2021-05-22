%% Load data
clear
clc
close all
centroids_pi4_all_ext = readmatrix('_conv_pi4_AllCenters_ext.csv');
centroids_pi4_all_int = readmatrix('_conv_pi4_AllCenters_int.csv');
centroids_pi_all_ext = readmatrix('_conv_pi_AllCenters_ext.csv');
centroids_pi_all_int = readmatrix('_conv_pi_AllCenters_int.csv');

centroids_pi4_ext_alu = readmatrix('A_pi4_Centers_ext.csv');
centroids_pi_ext_alu = readmatrix('A_pi_Centers_ext.csv');


centroids_pi4_all_int_alu = readmatrix('_conv_pi4_AllCenters_int.csv');
centroids_pi_all_ext_alu = readmatrix('_conv_pi_AllCenters_ext.csv');
centroids_pi_all_int_alu = readmatrix('_conv_pi_AllCenters_int.csv');


%%
%% Calculate ground truth
r = 31*10^(-3);
x_GT_pi4 = 0;
y_GT_pi4 = sin(pi/4)*r;
z_GT_pi4 = cos(pi/4)*r;

x_GT_pi = 0;
y_GT_pi = 0;
z_GT_pi = r;


%% Calculate errors

x_error_conv_pi4_all = centroids_pi4_all_ext(:,1) - x_GT_pi4;
y_error_conv_pi4_all = centroids_pi4_all_ext(:,2) - y_GT_pi4;
z_error_conv_pi4_all = centroids_pi4_all_ext(:,3) - z_GT_pi4;

%% conv_pi_all
x = 1:3;
data_mean = 1000*[mean(x_error_conv_pi4_all); mean(y_error_conv_pi4_all); mean(z_error_conv_pi4_all)];
stds = [std(x_error_conv_pi4_all), std(y_error_conv_pi4_all), std(z_error_conv_pi4_all)];

bar(x, abs(data_mean))
hold on
er = errorbar(x, abs(data_mean), stds, stds)
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

xlabel('Axis')
xticklabels({'X', 'Y', 'Z'})
ylabel('Mean error [mm]')
set(gca,'FontSize',18)

%%
figure(2)
N = 12042;
plot(centroids_pi_ext_alu(8*N:9*N,1), 'r', 'LineWidth', 2)
hold on
plot(centroids_pi_ext_alu(8*N:9*N,2), 'g', 'LineWidth', 2)
hold on
plot(centroids_pi_ext_alu(8*N:9*N,3), 'b', 'LineWidth', 2)
hold on
yline(x_GT_pi, 'r--')
hold on
yline(y_GT_pi, 'g--')
hold on
yline(z_GT_pi, 'b--')
hold on

xlabel('Measurement')
ylabel('Coordinate [m]')
legend('X', 'Y', 'Z', 'X ground truth', 'Y ground truth', 'Z ground truth')
set(gca,'FontSize',18)

%%
Data = centroids_pi_ext_alu(:, 1:3);
data = sqrt(sum(Data.^2,2));
plot(data)
%ylim([0 0.05])

