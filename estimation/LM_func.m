function [P, error] = LM_func(x0, xdata) 
    % Original
    syms mu_s mu_c v0 nabla F_n v F_fric
    
    sigma = 0.1;
    v1 = 2;
    E = mu_c + (mu_s - mu_c)*exp(-(v/v0)^2) + nabla*(v/F_n) - F_fric/F_n;
    jac = jacobian(E, [mu_s, mu_c, v0, nabla]);
    
    error = zeros(length(xdata),1);
    h = zeros(4,length(xdata));
    P = zeros(length(xdata),4);
    P(1,:) = x0;
    
    %iteration 1
    i=1;
    error(i) = [P(i,2) + (P(i,1) - P(i,2))*exp(-(xdata(i,2)/P(i,3))^2) + P(i,4)*(xdata(i,2)/xdata(i,1)) - xdata(i,3)/xdata(i,1)]; 
    G = double(subs(jac, [mu_s mu_c v0 nabla F_n v F_fric], [P(i,1), P(i,2), P(i,3), P(i,4),xdata(i,1),xdata(i,2),xdata(i,3)]));
    h(:,i) = -pinv([G'*G + sigma*diag(G'*G)])*G'*error(i);
    P(i+1,:) = P(i,:) + h(:,i)';
    
    for i =2:length(xdata)-1
        
        error(i) = [P(i,2) + (P(i,1) - P(i,2))*exp(-(xdata(i,2)/P(i,3))^2) + P(i,4)*(xdata(i,2)/xdata(i,1)) - xdata(i,3)/xdata(i,1)]; 
       
        G = double(subs(jac, [mu_s mu_c v0 nabla F_n v F_fric], [P(i,1), P(i,2), P(i,3), P(i,4),xdata(i,1),xdata(i,2),xdata(i,3)]));
        h(:,i) = -pinv([G'*G + sigma*diag(G'*G)])*G'*error(i);
        P(i+1,:) = P(i,:) + h(:,i)';
        
        %sigma
        delta = (error(i) - error(i-1));
        if delta > 0
            sigma = sigma * max(1/3, 1-(2*delta-1)^3);
            v1 = 2;
        else
            sigma = sigma * v1;
            v1 = v1 * 2;
        end
        
    end    



% Constrained
%     syms mu_s mu_c F_n v F_fric
%     nabla = 0;
%     v0 = 0.0014;
%     
%     
%     sigma = 0.1;
%     E = mu_c + (mu_s - mu_c)*exp(-(v/v0)^2) + nabla*(v/F_n) - F_fric/F_n;
%     jac = jacobian(E, [mu_s, mu_c]);
%     
%     error = zeros(length(xdata),1);
%     P = zeros(length(xdata),2);
%     P(1,:) = x0;
%     
%     for i =1:length(xdata)-1
%         error(i) = [P(i,2) + (P(i,1) - P(i,2))*exp(-(xdata(i,2)/v0)^2) + nabla*(xdata(i,2)/xdata(i,1)) - xdata(i,3)/xdata(i,1)]; 
%         G = double(subs(jac, [mu_s mu_c v0 nabla F_n v F_fric], [P(i,1), P(i,2), v0, nabla,xdata(i,1),xdata(i,2),xdata(i,3)]));
%         h = -pinv([G'*G + sigma*diag(G'*G)])*G'*error(i);
%         P(i+1,:) = P(i,:) + h';
%     end 
end