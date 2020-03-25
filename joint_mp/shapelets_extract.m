function [shapelets_a,topi_a_hist] = shapelets_extract(sinus,p_a,w,k)
    shapelets_a = zeros(k,w);
    t_a = numel(sinus);
    p_a_t = p_a;
    topi_a_hist = [];
    for id = 1:k
        [~, topi_a] = max(p_a_t);
        sh_range = topi_a:topi_a+w-1;
        shapelets_a(id,:) = sinus(sh_range);
        topi_a_hist = [topi_a_hist topi_a];
        p_a_t(max(1,topi_a-w+1):min(topi_a+w-1,t_a)) = 0; 
%         hold on
%         plot(sh_range,shapelets_a(id,:))
    end    
end
    
