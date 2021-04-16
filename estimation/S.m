function y = S(p)
dz = 0;
r = 31 * 10^-3;
%S = @(p)p(1)^2+p(2)^2+(p(3)-dz)^2-r^2

y = p(1)^2+p(2)^2+(p(3)-dz)^2-r^2;


end