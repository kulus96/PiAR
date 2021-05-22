%% Load data
clear
clc
close all
coeffs_alu = readmatrix('coeff_Data_Alu.npy.csv');
coeffs_pap = readmatrix('coeff_Data_Pap.npy.csv');
coeffs_all = readmatrix('coeff_Data_All.npy.csv');

%%
figure(1)
scatter(coeffs_alu(:,1), coeffs_alu(:,2), 'b')
hold on
scatter(coeffs_pap(:,1), coeffs_pap(:,2), 'r', 'Filled')
ylabel('Dynamic friction coefficient')
xlabel('Static friction coefficient')
legend( 'Aluminium', 'Cardboard', 'Location', 'SouthEast')
set(gca,'FontSize',18)
%title('Estimated friction coefficients vs. ground truth')

%%
figure(2)
scatter(coeffs_all(:,1), coeffs_all(:,2))
%%
figure()
plot(coeffs_alu(:,2), 'b')
hold on
plot(coeffs_pap(:,2), 'r')


