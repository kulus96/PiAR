clc
%clear
close all

load('quarter_circle.mat')
global theta_tm1 data g_t1m1 g_t2m1 %Global variables  that is updated
global  v thres % Design parameters

v = 1;

thres = 1;

data = [];
g_t1m1 = 0;
g_t2m1 = 0;
theta_tm1 = 0;


%simulation = sim('finger.slx',10);

clc
v = 10;
thres = 50;

close all;
res_CUSUM = [];

%test_data = simulation.test;
%test_data= sum(abs(test_data),2);

out = sim('finger', time_simulation);
%%
torque_sensor = out.Torque; %out.yout{1}.Values.Data;
force_sensor = out.Force; %out.yout{2}.Values.Data;
test_data = sum(abs(force_sensor),2);
%%

% for i = 1 : size(test_data,1)
%     
%     [res_CUSUM(i,:),detection(i,:)] = CUSUM_func(test_data(i));
%     
% end


% 
% for i = 1 : size(test_data,1)
%     
%     detection(i,:) = CUSUM_func(test_data(i));
%     
% end


for i = 1 : size(test_data,1)
    
    detection(i) = cusum_matlab(test_data(i));
    
end

%
figure(1);
plot(test_data)

%figure(2)
hold on;
%plot(res_CUSUM(:,:))
plot(detection(:)*50,'*')
hold off;

figure(3)
hold on
cusum(test_data)


