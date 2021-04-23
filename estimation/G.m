function y = G(x0, xdata)
%g = @(x0, xdata)[cross(xdata(1:3),x0(1:3)) + x0(4)*gradient(S(x0(1:3))) - xdata(4:6); S(x0(1:3))]; % Objective fucntion for moments and dist to surface
y = [cross(xdata(1:3),x0(1:3)) + x0(4)*gradient(S(x0(1:3))) - xdata(4:6); S(x0(1:3))];


end