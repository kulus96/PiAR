function [F_normal, F_friction, F_friction2] = estimateFriction(P_c, F, S)
    syms x y z
    p_vec= [x,y,z]';
    Q = double(subs(gradient(S(p_vec)),[x,y,z],[P_c(1), P_c(2),P_c(3)]));
    a = (Q'*F)/(Q'*Q);
    F_normal = a*Q;
    cos_theta = norm(F_normal)/(norm(F));
    F_friction = (sqrt(1-cos_theta^2)/cos_theta)*F_normal;

    F_friction2 = F-F_normal;
end