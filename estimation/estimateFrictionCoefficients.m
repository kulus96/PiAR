function [mu_static, mu_dynamic, stribeck, nabla] = estimateFrictionCoefficients(forces_tcp, moments_tcp, vel_tcp)

    %% Estimate contact point
    % Surface parameters
    R = 31 * 10^-3; % [m]
    A = eye(3);
    A(3,3)=1;
    dz = 0;

    %syms x y z k;
    %p = [x,y,z]';
    r = R;
    
    F_normals = zeros(length(forces_tcp),1);
    F_frictions = zeros(length(forces_tcp),1);

    % Functions for the numerical method
    %S = @(p)p(1)^2+p(2)^2+(p(3)-dz)^2-r^2; % Function for sphere surface
    %g = @(x0, xdata)[cross(xdata(1:3),x0(1:3)) + x0(4)*gradient(S(x0(1:3))) - xdata(4:6); S(x0(1:3))]; % Objective fucntion for moments and dist to surface
    
    
    options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt', 'Display', 'off');
    
    for i = 1:length(forces_tcp)
        f = forces_tcp(:,i);
        m = moments_tcp(:,i);

        % Analytical solution on a sphere
        %fm_ratio(i)= f'*m;
        %tic
        [c_sph, k_sph] = contactCentroidEstimation(A, R, f, m);
      
        % Numerical solution using the Levenberg-Marquardt method
        ydata = zeros(4,1);
        x0 = [[0;0;10.0]; k_sph]; % Initial guess
        xdata = [f;m];
        
        k_sph;
        G(x0, xdata);
        
        
        %[x,~,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(g,x0,xdata, ydata, [], [], options);
        [x,~,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(@G,x0,xdata, ydata, [], [], options);
        
        % Estimate friction force
        P_c = [x(1),x(2),x(3)]'
        %[F_normal, F_friction, F_friction2] = estimateFriction(P_c,f,S);
        [F_normal, F_friction, F_friction2] = estimateFriction(P_c,f)
        F_normals(i) = norm(F_normal);
        F_frictions(i) = norm(F_friction);
    end

    %% Determine start index
    diffs = diff(F_frictions);
    bool = true;
    cnt = 1;
    while abs(diffs(cnt)) > 0.01
        cnt = cnt + 1;
    end

    while diffs(cnt) <= 0.1      
        cnt = cnt + 1;
    end
    min_idx = cnt;
    while diffs(cnt) > 0.01
        cnt = cnt + 1;
    end
    max_idx = cnt;

    %% Estimate friction coefficients
    x = [0 0 0 0]';
    for n_start = min_idx:max_idx
        bool = true;
        n_data = 400;
        xdata = [F_normals(n_start:n_start+n_data,:), vel_tcp(n_start:n_start+n_data), F_frictions(n_start:n_start+n_data,:)];
        ydata = zeros(length(xdata),1);

        options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective', 'FunctionTolerance', 1e-8, 'Display', 'off');

        while bool
            x0 = rand(4,1);
            lb = [0.1 0.1 0 0];       % lower bound
            ub = [2 2 10 0.1];        % upper bound

            [x_, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(@g_func, x0, xdata, ydata, lb, ub, options);

            if length(residual) > 1
                if logical(output.firstorderopt < 1e-03)
                    bool = false;
                end
            end

        end
        x = x + x_;
    end
    x = x/(max_idx-min_idx+1);
    mu_static = x(1);
    mu_dynamic = x(2);
    stribeck = x(3);
    nabla = x(4);

end