function [force, torque, vel, friction_GT, normal_GT, mu_static, mu_dyn, label] = loadRawData(file_name,down_sample)
	load(file_name);
	force = data.force(1:down_sample:end,:)';
	torque = data.torque(1:down_sample:end,:)';
	vel = data.vel(1:down_sample:end,:);
	friction_GT = data.friction_GT(1:down_sample:end,:);
	normal_GT  =data.normal_GT(1:down_sample:end,:);
	mu_static = data.mu_static;
	mu_dyn = data.my_dyn;
	label = data.label;
end
