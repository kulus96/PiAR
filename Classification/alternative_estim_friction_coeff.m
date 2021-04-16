clear; clc;
addpath("./../estimation")
% load data
files = dir('../estimation/rawData/*.mat');

FORCES=[];
TORQUES=[];
VEL =[];
MU_STATIC = [];
MU_DYNAMIC =[];
LABEL = [];

% Training set
for i = 1: size(files,1)-1
	[force, torque, vel, ~, ~, mu_static, mu_dyn, label]= loadRawData(strcat(files(i).folder,'/', files(i).name), 1, true);
	l = int16(length(force)/6);
	% removes the first and last 1/6  of obtained data
	% finger 'should' move in this "section"
	FORCES = [FORCES; force(:,l:end-l)];
	TORQUES = [TORQUES; torque(:,l:end-l)];
	VEL = [VEL, vel(l:end-l)];
	LABEL = [LABEL,label];
	%scalar values
	MU_STATIC = [MU_STATIC; mu_static];
	MU_DYNAMIC = [MU_DYNAMIC; mu_dyn];
end
% test set
[force, torque, vel, ~, ~, mu_static, mu_dyn]= loadRawData(strcat(files(size(files,1)).folder,'/', files(size(files,1)).name), 1, true);
l = int16(length(force)/6);
test_FORCES=force(:,l:end-l);
test_TORQUES=torque(:,l:end-l);
test_VEL =vel(l:end-l);
test_MU_STATIC = mu_static;
test_MU_DYNAMIC =mu_dyn;

%% Create windows
window_size = 20;
number_of_windows = 50;
train_MU_STATIC = [];
train_MU_DYNAMIC = [];
train_FORCES = [];
train_TORQUES = [];
train_VEL = [];
train_data_input = [];
train_data_output = [];
label_data = [];
length_of_data = length(FORCES);
for i = 1: size(files,1)-1
	for window_index = 1 : number_of_windows

        train_data_output = [train_data_output; [MU_STATIC(i), MU_DYNAMIC(i)]];
		label_data = [label_data, LABEL(i)];
		random_index = randi([1,length_of_data-window_size-1]);
		train_FORCES_temp =  FORCES((i*3)-2:i*3,random_index:random_index+window_size-1);
		train_TORQUES_temp = TORQUES((i*3)-2:i*3,random_index:random_index+window_size-1);
		train_VEL_temp = VEL(random_index:random_index+window_size-1,i);
		temp_data =  [reshape(train_FORCES_temp,[],1); reshape(train_TORQUES_temp,[],1); train_VEL_temp];
		train_data_input =  [train_data_input temp_data];
	end
end
train_data_output = train_data_output';



%load('network_dense.mat')
%surface_prop_nn = net(train_data_input)';


%% Neural net Dense
% inputs = train_data_input;
% targets = train_data_output;
%
% % Create a Fitting Network
% hiddenLayerSize = [7 4];
% net = fitnet(hiddenLayerSize);
%
% % Set up Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
%
% % Train the Network
% [net,tr] = train(net,inputs,targets);%, 'useGPU', 'yes');
%
% % Test the Network
% outputs = net(inputs);
% errors = gsubtract(outputs,targets);
% performance = perform(net,targets,outputs);
%
% % View the Network
% %view(net)
%
% % Plots
% % Uncomment these lines to enable various plots.
% % figure, plotperform(tr)
% % figure, plottrainstate(tr)
% % %figure, plotfit(targets,outputs)
% % figure, plotregression(targets,outputs)
% % figure, ploterrhist(errors)
% %%
%
% random_index = randi([1,length(test_TORQUES)-window_size-1]);
% test2_FORCES_temp =  test_FORCES(:,random_index:random_index+window_size-1);
% test2_TORQUES_temp = test_TORQUES(:,random_index:random_index+window_size-1);
% test2_VEL_temp = test_VEL(random_index:random_index+window_size-1,:);
% temp_data =  [reshape(train_FORCES_temp,[],1); reshape(train_TORQUES_temp,[],1); train_VEL_temp];
%
% truth = [test_MU_STATIC, test_MU_DYNAMIC]
% dense_nn_res = net(temp_data)'

% end of neural net
%% Multivariate Linear regression model
% NxK where N = number of observations, K number of regression coeff

X = [ones([size(train_data_input,2),1]), train_data_input'];
Y = train_data_output';

beta = mvregress(X, Y);
MLRM_res = [1 temp_data']*beta;
surface_prop_mlrm = ([ones(length(train_data_input),1) train_data_input']*beta);
