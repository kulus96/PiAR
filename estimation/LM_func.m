function [P] = g_func(x0, xdata) 
    syms mu_s mu_c v0 nabla F_n v F_fric
    P = x0;
    sigma = 0.01;

    E = mu_c + (mu_s - mu_c)*exp(-(v/v0)^2) + nabla*(v/F_n) - F_fric/F_n;
    jac = jacobian(E, [mu_s, mu_c, v0, nabla]);
    % x0 = [mu_s, mu_c, v0, nabla]
    % xdata = [F_n v F_fric]
    %g_mat = zeros(length(xdata),1);
    
    for i =1:10%length(xdata)
        error = double(subs(E, [mu_s mu_c v0 nabla F_n v F_fric], [x0(1), x0(2), x0(3), x0(4),xdata(i,1),xdata(i,2),xdata(i,3)]));
        G = subs(jac, [mu_s mu_c v0 nabla F_n v F_fric], [x0(1), x0(2), x0(3), x0(4),xdata(i,1),xdata(i,2),xdata(i,3)]);
        h = inv([G'*G + sigma*diag(G'*G)])*G'*error;
        P = P + h;
    end
    display(double(P))
end