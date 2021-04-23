function [force, torque, vel, friction_GT, normal_GT, mu_static, mu_dyn, label] = loadRawData(file_name,down_sample, with_noise)
	if nargin<3
		disp('Noise = True')
		with_noise=True;
	else
		with_noise = boolean(with_noise);
	end

	load(file_name);
	vel = data.vel(1:down_sample:end,:);
	friction_GT = data.friction_GT(1:down_sample:end,:);
	normal_GT  =data.normal_GT(1:down_sample:end,:);
	mu_static = data.static_fric_coeff;
	mu_dyn = data.dynamic_fric_coeff;
	label = data.label;
	force = data.force(1:down_sample:end,:)';
	torque = data.torque(1:down_sample:end,:)';

	mean = 0;
	std = 0.003;

	if with_noise
		% with noise on forceTorque sensor
		force = force + normrnd(mean, std, size(force,1),size(force,2));
		torque = torque + normrnd(mean, std, size(torque,1), size(torque,2));
	end
end
