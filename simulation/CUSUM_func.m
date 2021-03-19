% Page 68 in the book Adaptive filtering and change detection
% Algorithm 3.3
% CUSUM LS FILTER

function  detection = CUSUM_func(in)
    global theta_tm1 data g_t1m1 g_t2m1 %Global variables  that is updated
    global  v thres % Design parameters
    
    detection = 0.0;
    theta_t = zeros(1, size(in, 2));
    if(size(data,2) == 0)
        data = in;
        g_t1m1 = 0;
        g_t2m1 = 0;
        theta_tm1 = 0;
    else
        
        theta_t = mean(data, 1); %1/size(data,2)*sum(data);
        epsilon = in - theta_tm1;

        s_t1 = epsilon;
        s_t2 = epsilon * (-1);

        g_t1 = max(g_t1m1 + s_t1 - v,0);
        g_t2 = max(g_t2m1 + s_t2 - v,0);

        % Check for changes
        %for i = 1 : size(in, 2)
            if (logical(g_t1 > thres) | logical(g_t2 > thres))
                detection = 1.0;
                data = [];
                g_t1 = zeros(1, size(in, 2));
                g_t2 = zeros(1, size(in, 2));
                %break;
            else 
                detection = 0.0; 
            end
        %end
        

        % Update variables
        if detection ~= 1.0
            data = [data; in];
            theta_tm1 = theta_t;
            g_t1m1 = g_t1;
            g_t2m1 = g_t2;
        end
    end

end

