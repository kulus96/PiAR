function [c_sphere, K_sphere] = contactCentroidEstimation(A,R, f, m)

% Sphere force centroid estimation_ single Old Paper
sig_prime = norm(m)^2-R^2*norm(f)^2;
K_sphere = -sign(f'*m)/(sqrt(2)*R)*sqrt(sig_prime + sqrt(sig_prime^2+4*R^2*(f'*m)^2));

    % Check for f'*m = 0
    if K_sphere > 0
        c_sphere = (1)/(K_sphere*(K_sphere^2+norm(f)^2))*(K_sphere^2*m+K_sphere*cross(f,m)+(f'*m)*f);
    else
        r_0= (cross(f,m))/(norm(f)^2);
        f_prime = A*f;
        r_0_prime = A*r_0;
        lambda = (-f_prime'*r_0_prime- sqrt((f_prime'*r_0_prime)^2-norm(f_prime)^2*(norm(r_0_prime)^2-R^2)))/norm(f_prime)^2;
        c_sphere = r_0 + lambda*f;       
    end

end

