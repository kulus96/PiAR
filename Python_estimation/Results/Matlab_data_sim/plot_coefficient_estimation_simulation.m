%% Load data
clear
clc
close all
coeffs = readmatrix('estimation_coeff_results.csv');
coeffs_noise = readmatrix('estimation_coeff_noise_results.csv');
labels = readmatrix('estimation_labels_results.csv');
means = readmatrix('classification_means_results.npy.csv');
means_noise = readmatrix('classification_means_noise_results.npy.csv');
set(gca,'FontSize',20)
%% No noise CLassifier
%figure(1)
%bar(means*100)
%xticklabels({"GNB", "SVM poly", "SVM linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"})
%xtickangle(45)
%ylabel('Accuracy [%]')

figure(2)
new_means = [means(1); means(5); means(8); means(10)];
bar(new_means*100)
%xticklabels({"GNB", "SVM poly", "SVM linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"})
xticklabels({"GNB", "LDA", "RandomForest", "MLP"})
xtickangle(45)
ylabel('Accuracy [%]')
title('Performance of different classifiers')

%% With noise Classifier
%figure(1)
%bar(means_noise*100)
%xticklabels({"GNB", "SVM poly", "SVM linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"})
%xtickangle(45)
%ylabel('Accuracy [%]')

figure(2)
new_means_noise = [means_noise(1); means_noise(5); means_noise(8); means_noise(10)];
bar(new_means_noise*100)
%xticklabels({"GNB", "SVM poly", "SVM linear", "KNN", "LDA", "DTC", "AdaBoost", "RandomForest", "GPC", "MLP"})
xticklabels({"GNB", "LDA", "RandomForest", "MLP"})
xtickangle(45)
ylabel('Accuracy [%]')
%title('Performance of different classifiers')
%new_means_noise_str = [num2str(means_noise(1))+str('%'); numstr(means_noise(5)); num2str(means_noise(8)); num2str(means_noise(10))];

%text(1:length(new_means_noise), 100*[new_means_noise(2), new_means_noise(2), new_means_noise(2), new_means_noise(2)], num2str(round(new_means_noise*100,2)), 'vert','bottom','horiz','center', 'FontSize', 16)
text(1:length(new_means_noise), 100*[new_means_noise(2), new_means_noise(2), new_means_noise(2), new_means_noise(2)], num2str(round(new_means_noise*100,2)), 'vert','bottom','horiz','center', 'FontSize', 16)

set(gca,'FontSize',18)

%% No noise coefficients
figure(3)
mu_static_GT = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10, 0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61];
mu_dyn_GT =    [0.42, 0.57, 0.95, 0.47, 0.36, 0.53, 0.44, 0.21, 0.29, 1.40, 0.40, 0.48, 0.32, 0.52];
scatter(coeffs(:,1), coeffs(:,2), 'b')
hold on
scatter(mu_static_GT, mu_dyn_GT, 'r')
ylabel('Dynamic friction coefficient')
xlabel('Static friction coefficient')
legend( 'Estimation', 'Ground truth')
title('Estimated friction coefficients vs. ground truth')

%% With noise coefficients
figure(4)
mu_static_GT = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10, 0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61];
mu_dyn_GT =    [0.42, 0.57, 0.95, 0.47, 0.36, 0.53, 0.44, 0.21, 0.29, 1.40, 0.40, 0.48, 0.32, 0.52];
scatter(coeffs_noise(:,1), coeffs_noise(:,2), 'b')
hold on
scatter(mu_static_GT, mu_dyn_GT, 'r', 'filled')
ylabel('Dynamic friction coefficient')
xlabel('Static friction coefficient')
legend( 'Estimation', 'Ground truth')
title('Estimated friction coefficients vs. ground truth')
set(gca,'FontSize',18)


%% Plot Stribeck Bar
stribecks = coeffs(:,3);
iter = length(coeffs)/10;
stribeck_means = zeros(iter, 1);
stribeck_stds = zeros(iter, 1);
stribeck_data = zeros(iter, 10);


for i = 0:iter-1
    stribeck_means(i+1) = mean(stribecks(1+i*10:(i+1)*10));
    stribeck_stds(i+1) = std(stribecks(1+i*10:(i+1)*10));
end

figure(8)
x = 1:iter;
bar(x, stribeck_means)
hold on
er = errorbar(x, stribeck_means, stribeck_stds, stribeck_stds);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

xlabel('Material')
%xticklabels({'X', 'Y', 'Z'})
ylabel('Mean stribeck velocity')
set(gca,'FontSize',18)

figure(9)
for i = 0:iter-1
    stribeck_data(i+1, :) = stribecks(1+i*10:(i+1)*10);
end

boxplot(stribeck_data')

%% Plot viscosity
visc = coeffs(:,4);
iter = length(coeffs)/10;
visc_means = zeros(iter, 1);
visc_stds = zeros(iter, 1);
visc_data = zeros(iter, 10);


for i = 0:iter-1
    visc_means(i+1) = mean(visc(1+i*10:(i+1)*10));
    visc_stds(i+1) = std(visc(1+i*10:(i+1)*10));
end

figure(8)
x = 1:iter;
bar(x, visc_means)
hold on
er = errorbar(x, visc_means, visc_stds, visc_stds);
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

xlabel('Material')
%xticklabels({'X', 'Y', 'Z'})
ylabel('Mean visc')
set(gca,'FontSize',18)

figure(9)
for i = 0:iter-1
    visc_data(i+1, :) = visc(1+i*10:(i+1)*10);
end

boxplot(visc_data')
