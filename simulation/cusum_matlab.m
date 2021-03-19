function detection = cusum_matlab(input)
    global data;
    
    if(size(data, 2) == 0)
       data = input;
       detection = 0;
    else
        data = [data; input];
        %size(data)
        [U, L] = cusum(data);
        
        if( U ~= 0 | L ~= 0)
            detection = 1;
            data = [];
        else
            detection = 0;
        end
    end
end