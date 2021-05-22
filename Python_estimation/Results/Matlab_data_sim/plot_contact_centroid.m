%% Load data
clear
clc
close all
centroids = readmatrix('contact_centroid_sim.npy.csv');
K_values = readmatrix('contact_centroid_K_values.npy.csv');
centroids_all = readmatrix('contact_centroids_all_data.npy.csv');
%% Calculate ground truth
angle = pi/4;
r = 31*10^(-3);
x_GT = sin(pi/4)*r;
y_GT = 0;
z_GT = cos(pi/4)*r;
%% Calculate errors
x_error = centroids(:,1) - x_GT;
y_error = centroids(:,2) - y_GT;
z_error = centroids(:,3) - z_GT;
x_error_all = centroids_all(:,1) - x_GT;
y_error_all = centroids_all(:,2) - y_GT;
z_error_all = centroids_all(:,3) - z_GT;
%% Plot centroids
plot(centroids(:,1), 'r')
hold on
plot(centroids(:,2), 'g')
hold on
plot(centroids(:,3), 'b')

%% Plot centroids
plot(centroids_all(:,1), 'r')
hold on
plot(centroids_all(:,2), 'g')
hold on
plot(centroids_all(:,3), 'b')

%% Plot errors
plot(x_error, 'r')
hold on
plot(y_error, 'g')
hold on
plot(z_error, 'b')
%% Plot errors
plot(x_error_all, 'r')
hold on
plot(y_error_all, 'g')
hold on
plot(z_error_all, 'b')
xlim([0 1000])
%%
x = 1:3;
data_mean = [mean(x_error); mean(y_error); mean(z_error)]*1000;
stds = [std(x_error_all), std(y_error_all), std(z_error_all)]*1000;

data = [x_error; y_error; z_error];
bar(x, abs(data_mean))
hold on
er = errorbar(x, abs(data_mean), stds, stds, 'LineWidth', 1.2, 'CapSize', 14)
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

%xlabel('Axis')
xticklabels({'X-axis', 'Y-axis', 'Z-axis'})
ylabel('Mean error [mm]')
set(gca,'FontSize',18)

