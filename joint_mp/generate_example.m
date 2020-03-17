% consensus motif example for a sequence of 10 time series
% Commented out sections were used to generate the file random_walk.txts

% Generate a number of short time series.
rng(0, 'twister');
k = 10;
baselen = 2^12;
L = [2^12; 2^10; 2^10; 2^11 + 1; 2^12 + 76; 2^11 + 130; 2^11 + 60; 2^11 + 80; 2^10 + 100; 2^11 + 100];
subsequence_len = 301;

T = zeros(sum(L) + k - 1,1);
T(1:L(1)) = cumsum(randn(L(1), 1));
j = L(1) + 1;
for i = 2 : k
    T(j) = NaN;
    T(j + 1 : j + L(i)) = cumsum(randn(L(i), 1));
    j = j + 1 + L(i);
end

figure();
ax = axes();
j = 1;
hold on;
for i = 1 : 10
    plot(zscore(T(j : j + L(i) - 1),1) + 5*i);
    j = j + 1 + L(i);
end
hold off;
title(sprintf('k = %d time series',k));
ax.YTick = [];
drawnow;
[sol,obj] = consensus_search.from_nan_cat(T,subsequence_len,true);
% title(sprintf('corresponding consensus motif for subsequence length: %d radius %g',subsequence_len,sol.radius));
% drawnow;

% Example of the same thing using the constructor interface directly
% True forces it to generate a basic plot

% obj_s = consensus_search(obj.ts, subsequence_len);
% sol_s = obj_s.solve_opt(true);

% Example of  "k of P"
% Here these are just spurious correlations, if we look for the best match
% in 7 out of 10, the distribution is considerably tighter.

% sol_k = obj.solve_optimal_subset(7, true);


% You can also use 
% title_string = 'demo';
% obj_s.plot_dist(sol, title_string);

% optional to test against brute force version
% On this particular problem execution time doesn't differ that much.
% This may be updated to bundle a C++ implementation of mpx_AB to make the
% difference more apparent.


% [sol_0] = obj.solve_naive(true);
