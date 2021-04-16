% This File contains the classification functionality:
% KNN, NaiveBayes, SVM and NN

% retrieve estimated surface properties



%% KNN
label = ["hard steel on hard steel","Mild steel on mild steel", "Mild steel on lead", "Aluminum on mild steel", "Copper on mild steel","Nickel on nickel", "Brass on mild steel","Zinc on cast iron", "Copper on cast iron",...
 "Aluminum on aluminum", "Glass on glass", "Oak on oak (parallel to grain)", "Oak on oak (perpendicular)","Leather on oak (parallel)"];
mu_s = [0.78, 0.74, 0.95, 0.61, 0.53, 1.10 ,0.51, 0.85, 1.05, 1.05, 0.94, 0.62, 0.54, 0.61];
mu_c = [0.42, 0.57, 0.95, 0.47, 0.36, 0.53 ,0.44, 0.21, 0.29, 1.40, 0.40, 0.48, 0.32, 0.52];

stribeck_coef = zeros(1,length(mu_c)); % should be set to the correct values when a working estimator is avaliable.
viscosity_coef = zeros(1,length(mu_c));
%surface_properties = [surface_prop_nn, zeros(length(surface_prop_nn),2)];
surface_properties = [surface_prop_mlrm, zeros(length(surface_prop_mlrm),2)];

classes = [mu_s', mu_c', stribeck_coef', viscosity_coef'];
[idx, list] = knnsearch(classes, surface_properties);
class_guess = label(idx);
performance = class_guess == label_data;
tp = sum(performance);
fp = length(performance)-tp;
accuracy = tp/length(performance)

%%
X = surface_properties;

% setup Y
Y=[];
for i = 1:number_of_windows:length(X)
	Y = [Y; repmat(find(label==label_data(i)),number_of_windows,1)];
end

Mdl = fitcecoc(X,Y);
%%
classificationLearner(X,Y);

% classificationLearner(X,Y) opens the Classification Learner app and populates
% the New Session from Arguments dialog box with the n-by-p predictor matrix X
% and the n class labels in the vector Y. Each row of X corresponds to one observation,
% and each column corresponds to one variable. The length of Y and the number of
% rows of X must be equal.
%% NN
