classdef consensus_search < handle
    % version 5
    properties
        ts;
        sublen;
        mu;
        invn;
        mp;
        mpi;
        ffts;
        tsCount;
        joinsAvailable;
        base;
    end
    methods
        function obj = consensus_search(ts, sublen)
            [valid, msg] = consensus_search.validateParameters(ts, sublen);
            if ~valid
                error(msg);
            end
            % takes a cell array of time series,
            %       a subsequence length,
            %       and the minimum number of time series which a candidate
            %       subsequence must match against
            %       Since this is a work in progress pending agreement on a couple definitions,
            %       we currently only support the case where a candidate
            %       must match against every time series
            
            N = length(ts);
            obj.ts = ts;
            obj.sublen = sublen;
            obj.mu = cell(N,1);
            obj.invn = cell(N,1);
            obj.mp = cell(N, 1);
            obj.mpi = cell(N,1);
            obj.ffts = cell(N, 1);
            counts = zeros(N, 1);
            obj.joinsAvailable = 0;
            obj.tsCount = length(obj.ts);
            obj.base = struct(...
                'nearest_neighbor_dists', repmat(2 * sqrt(obj.sublen), length(obj.ts),1),...
                'nearest_neighbor_indices', NaN(length(obj.ts), 1),...
                'radius', 2 * sqrt(obj.sublen));
            for i = 1 : N
                counts(i) = length(ts{i});
            end
            % in Matlab, padding to a power of 2 or splitting into chunks
            % is usually best. Here most of our time is typically dedicated
            % to joins.
            for i = 1 : size(ts,1)
                [obj.mu{i}, obj.invn{i}] = muinvn(obj.ts{i}, sublen);
                obj.ffts{i} = fft(obj.ts{i}, 2^nextpow2(length(obj.ts{i})));
                if any(isnan(obj.invn{i})) || any(isinf(obj.invn{i})) || any(isnan(obj.ffts{i})) || any(isinf(obj.ffts{i}))
                    % This could throw an index too
                    error('Either the input contains a constant section, which can''t be properly normaized, or we have run into very bad rounding. Either way check your input');
                end
            end
        end
        
        function solve_joins(obj, countper)
            if countper > length(obj.ts) - 1
                error('requested joins must not exceed one less than the number of time series');
            elseif obj.joinsAvailable >= countper
                return;
            else
                for i = 1 : length(obj.ts)
                    for j = obj.joinsAvailable + 1 : countper
                        if i + j > length(obj.ts)
                            [obj.mp{i,j}, obj.mpi{i,j}] = obj.ABjoin(i, i + j - length(obj.ts));
                        else
                            [obj.mp{i,j}, obj.mpi{i,j}] = obj.ABjoin(i, i + j);
                        end
                    end
                end
            end
            obj.joinsAvailable = countper;
        end
        
        function [mp] = pooled_min(obj, a, count)
            if count > size(obj.mp, 2)
                error('insufficient number of joins have been precomputed');
            end
            mp = obj.mp{a,1};
            for i = 2 : count
                % Here we only need to indicate where we found the
                % original, so we know which profile indices to check later
                mp(obj.mp{a,i} < mp) = obj.mp{a,i}(obj.mp{a,i} < mp);
            end
        end
        
        function bsf = solve_optimal_subset(obj, minContain, plot_sol)
            if nargin == 2
                plot_sol = false;
            end
            if (minContain > obj.tsCount) || (minContain < 2)
                error('invalid choice of k');
                %commented out for testing
            elseif minContain == obj.tsCount
                bsf = obj.solve_opt(plot_sol);
                return;
            end
            mxExclude = obj.tsCount - minContain;
            obj.solve_joins(mxExclude + 1);
            bsf = obj.base;
            for i = 1:length(obj.ts)
                [p_mp] = obj.pooled_min(i, mxExclude + 1);
                [prof, S] = sort(p_mp);
                ordering = [(i + 1: length(obj.ts))'; (1 : i - 1)'];
                for j = 1 : length(prof)
                    if prof(j) > bsf.radius
                        break;
                    end
                    cand = obj.base;
                    cand.radius = 0;
                    cand.nearest_neighbor_dists(i) = 0;
                    cand.nearest_neighbor_indices(i) = S(j);
                    for k = 1 : mxExclude + 1
                        b = ordering(k); % Here b is a coordinate value. It maps from the ordering starting from A back to the original ordering
                        cand.nearest_neighbor_indices(b) = obj.mpi{i, k} (S(j));
                        cand.nearest_neighbor_dists(b) = obj.mp{i, k} (S(j));
                    end
                    query = (obj.ts{i}(S(j) : S(j) + obj.sublen - 1) - obj.mu{i}(S(j))) .* obj.invn{i}(S(j));
                    for k = mxExclude + 2 : length(obj.ts) - 1
                        b = ordering(k);
                        [dist, ind] = obj.findnearest(query, b);
                        cand.nearest_neighbor_dists(b) = dist;
                        cand.nearest_neighbor_indices(b) = ind;
                    end
                    mx = consensus_search.mink(cand.nearest_neighbor_dists, minContain);
                    cand.radius = mx(end);
                    if cand.radius < bsf.radius
                        bsf = cand;
                    end
                end
            end
            if plot_sol
                obj.plot_topk(bsf, minContain, sprintf('radius: %g', bsf.radius));
            end
        end
        
        function bsf = solve_opt(obj,plot_sol)
            if nargin == 1
                plot_sol = false;
            end
            if obj.joinsAvailable < 1
                obj.solve_joins(1);
            end
            bsf = obj.base;
            % assume for now the only AB join used is 1 ahead modulo count
            for i = 1 : obj.tsCount
                [prof, S] = sort(obj.mp{i,1});
                for j = 1:length(prof)
                    if prof(j) > bsf.radius
                        break;
                    end
                    cand = obj.base;
                    % AB join pair is implicitly (i, mod(i, number of time series) + 1)
                    A = i;
                    B = mod(i, obj.tsCount) + 1;
                    cand.nearest_neighbor_indices(A) = S(j);
                    cand.nearest_neighbor_indices(B) = obj.mpi{i,1}(S(j));
                    cand.nearest_neighbor_dists(A) = 0;
                    cand.nearest_neighbor_dists(B) = prof(j);
                    cand.radius = prof(j);
                    if cand.nearest_neighbor_indices(A) > length(obj.ts{A}) - obj.sublen + 1 || cand.nearest_neighbor_indices(B) > length(obj.ts{B}) - obj.sublen + 1
                        fprintf('problem %d %d \n',A,B);
                    end
                    q = (obj.ts{A}(S(j) : S(j) + obj.sublen - 1) - obj.mu{A}(S(j))) .* obj.invn{A}(S(j));
                    for k = 1 : obj.tsCount
                        if (k == A) || (k == B)
                            continue;
                        end
                        [dist, ind] = obj.findnearest(q, k);
                        if ind > length(obj.ts{k}) - obj.sublen + 1
                            fprintf('problem %d %d %d\n',A, B, k);
                        end
                        cand.nearest_neighbor_dists(k) = dist;
                        cand.nearest_neighbor_indices(k) = ind;
                        if dist > cand.radius
                            cand.radius = dist;
                        end
                        if(cand.radius > bsf.radius)
                            break;
                        end
                    end
                    if cand.radius < bsf.radius
                        bsf = cand;
                    end
                end
            end
            if plot_sol
                obj.plot_distribution(bsf, sprintf('radius: %g',bsf.radius));
            end
        end
        
        function bsf = solve_stamp(obj, use_naive, plot_sol)
            % This is an augmented implementation of stamp from [1]. It's here for performance comparisons.
            % We wanted to keep implementation details as uniform as possible, but you should generally only use it for comparative reasons.
            if nargin < 3
                plot_sol = false;
            end
            if nargin < 2
                use_naive = false;
            end
            bsf = obj.base;
            for i = 1 : obj.tsCount
                comps = cell(obj.tsCount, 1);
                compsi = cell(obj.tsCount, 1);
                for j = 1 : obj.tsCount
                    if i == j
                        continue;
                    end
                    if use_naive
                        [comps{j}, compsi{j}] = naive_stampab(obj.ts{i}, obj.ts{j}, obj.sublen);
                    else
                        [comps{j}, compsi{j}] = stampab(obj.ts{i}, obj.ts{j},obj.sublen);
                    end
                end
                % now compute radius
                cand_prof = zeros(length(obj.ts{i}) - obj.sublen + 1, 1);
                for j = 1 : obj.tsCount
                    if i == j
                        continue;
                    end
                    cand_prof(cand_prof < comps{j}) = comps{j}(cand_prof < comps{j});
                end
                [radius, nni] = min(cand_prof);
                if radius < bsf.radius
                    cand = obj.base;
                    cand.nearest_neighbor_indices(i) = nni;
                    cand.radius = radius;
                    cand.nearest_neighbor_dists(i) = 0;
                    for j = 1 : obj.tsCount
                        if i == j
                            continue;
                        end
                        cand.nearest_neighbor_dists(j) = comps{j}(nni);
                        cand.nearest_neighbor_indices(j) = compsi{j}(nni);
                    end
                    bsf = cand;
                end
            end
            if plot_sol
                obj.plot_distribution(bsf, sprintf('radius: %g',bsf.radius));
            end
        end
        
        function bsf = solve_stomp(obj, plot_sol)
            % This is an augmented implementation of stamp from [1]. It's here for performance comparisons.
            % We wanted to keep implementation details as uniform as possible, but you should generally only use it for comparative reasons.
            if nargin < 2
                plot_sol = false;
            end
            bsf = obj.base;
            for i = 1 : obj.tsCount
                comps = cell(obj.tsCount, 1);
                compsi = cell(obj.tsCount, 1);
                for j = 1 : obj.tsCount
                    if i == j
                        continue;
                    end
                    [comps{j}, compsi{j}] = mpx_AB(obj.ts{i}, obj.ts{j}, obj.sublen);
                end
                % now compute radius
                cand_prof = zeros(length(obj.ts{i}) - obj.sublen + 1, 1);
                for j = 1 : obj.tsCount
                    if i == j
                        continue;
                    end
                    cand_prof(cand_prof < comps{j}) = comps{j}(cand_prof < comps{j});
                end
                [radius, nni] = min(cand_prof);
                if radius < bsf.radius
                    cand = obj.base;
                    cand.nearest_neighbor_indices(i) = nni;
                    cand.radius = radius;
                    cand.nearest_neighbor_dists(i) = 0;
                    for j = 1 : obj.tsCount
                        if i == j
                            continue;
                        end
                        cand.nearest_neighbor_dists(j) = comps{j}(nni);
                        cand.nearest_neighbor_indices(j) = compsi{j}(nni);
                    end
                    bsf = cand;
                end
            end
            if plot_sol
                obj.plot_distribution(bsf, sprintf('radius: %g',bsf.radius));
            end
        end
        
        function bsf = solve_naive(obj, plot_sol)
            % brute force method for comparison, in order search + naively
            % computed convolutions. Similar to stamp, we wanted to keep
            % the implementation as uniform as possible for the purpose of comparison.
            if nargin < 2
                plot_sol = false;
            end
            bsf = obj.base;
            for i = 1 : obj.tsCount
                for j = 1 : length(obj.ts{i}) - obj.sublen + 1
                    cand = obj.base;
                    cand.nearest_neighbor_dists(i) = 0;
                    cand.nearest_neighbor_indices(i) = j;
                    q = (obj.ts{i}(j + obj.sublen - 1 : -1 : j) - obj.mu{i}(j)).* obj.invn{i}(j);
                    for k = 1 : obj.tsCount
                        if i == k
                            continue;
                        end
                        cv = conv2(obj.ts{k}, q, 'valid');
                        [cr,index] = max(cv .* obj.invn{k});
                        dist = sqrt(2 * obj.sublen * (1 - cr));
                        cand.nearest_neighbor_dists(k) = dist;
                        cand.nearest_neighbor_indices(k) = index;
                    end
                    cand.radius = max(cand.nearest_neighbor_dists);
                    if cand.radius < bsf.radius
                        bsf = cand;
                    end
                end
            end
            if plot_sol
                obj.plot_distribution(bsf, sprintf('radius: %g',bsf.radius));
            end
        end
        
        function [dist,index] = findnearest_naive(obj, q, tsmi)
            % This assumes q has been mean centered, so sum(q) is
            % approximately 0, also it has been reversed since we're using
            % direct convolution to compute correlation
            cv = conv2(obj.ts{tsmi}, q, 'valid');
            [cr,index] = max(cv .* obj.invn{tsmi});
            dist = sqrt(2 * obj.sublen * (1 - cr));
        end
        
        function [dist,index] = findnearest(obj, q, tsmi)
            % This assumes q has been mean centered, so sum(q) is approximately 0
            cv = ifft(obj.ffts{tsmi} .* conj(fft(q,length(obj.ffts{tsmi}))), 'symmetric');
            [cr,index] = max(cv(1 : length(obj.invn{tsmi})) .* obj.invn{tsmi});
            dist = sqrt(2 * obj.sublen * (1 - cr));
        end
        
        function plot_locations(obj, tsmi, normalize)
            % Todo: accept plotting keyword args
            if nargin == 2
                normalize = false;
            end
            fg = figure();
            ax = axes(fg);
            hold(ax,'on');
            st = zeros(length(tsmi),2);
            st(1, :) = [1, length(obj.ts{1})];
            for i = 2 : length(tsmi)
                st(i,:) = [st(i - 1, 1) + st(i - 1 ,2), length(obj.ts{i})];
            end
            for i = 1 : length(tsmi)
                if normalize
                    plot(ax, (st(i,1) : st(i,1) + st(i,2) - 1)', zscore(obj.ts{i},1));
                else
                    plot(ax, (st(i,1) : st(i,1) + st(i,2) - 1)', obj.ts{i}); %change made for primate plot
                end
            end
            for i = 1 : length(tsmi)
                if normalize
                    z = zscore(obj.ts{i},1); % changed for primate plot
                else
                    z = obj.ts{i};
                end
                plot(ax, st(i,1) + tsmi(i) - 1, z(tsmi(i)), 'kx', 'MarkerSize', 5); %for now this works
            end
            hold(ax,'off');
            drawnow;
        end
        
        function plot_topk(obj, bsf, k, heading)
            A = find(bsf.nearest_neighbor_dists == 0);
            figure();
            ax = axes();
            hold(ax, 'on');
            [~,mxi] = consensus_search.mink(bsf.nearest_neighbor_dists, k);
            mxi = sort(mxi); % we prefer to plot them in their original order. Unfortunately it still doesn't match colors compared to a full plot.
            for i = 1 : length(mxi)
                if mxi(i) == A
                    plot(ax, (obj.ts{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i)):bsf.nearest_neighbor_indices(mxi(i))+obj.sublen-1) - obj.mu{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i)))) .* obj.invn{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i))), 'LineWidth', 2);
                else
                    plot(ax, (obj.ts{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i)):bsf.nearest_neighbor_indices(mxi(i))+obj.sublen-1) - obj.mu{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i)))) .* obj.invn{mxi(i)}(bsf.nearest_neighbor_indices(mxi(i))));
                end
            end
            title(heading);
            hold(ax, 'off');
            drawnow;
        end
        
        function plot_distribution(obj, bsf, heading)
            % would need to be updated for kmin, assumes all time series
            % participate
            A = find(bsf.nearest_neighbor_dists == 0);
            figure();
            ax = axes();
            hold(ax, 'on');
            for i = 1 : obj.tsCount
                if i == A
                    plot(ax, (obj.ts{i}(bsf.nearest_neighbor_indices(i):bsf.nearest_neighbor_indices(i)+obj.sublen-1) - obj.mu{i}(bsf.nearest_neighbor_indices(i))) .* obj.invn{i}(bsf.nearest_neighbor_indices(i)), 'LineWidth', 2);
                else
                    plot(ax, (obj.ts{i}(bsf.nearest_neighbor_indices(i):bsf.nearest_neighbor_indices(i)+obj.sublen-1) - obj.mu{i}(bsf.nearest_neighbor_indices(i))) .* obj.invn{i}(bsf.nearest_neighbor_indices(i)));
                end
            end
            title(heading);
            hold(ax, 'off');
            drawnow;
        end
        
        function difference_plot(obj, dists, tsmi)
            % locate seed based on distance value of 0, if there's more than one, then they either
            % generate the same set or the set itself is not unique
            primary = find(dists == 0, 1);
            if isempty(primary)
                error('invalid distance set');
            end
            % difference plot
            fg = figure();
            ax = axes(fg);
            S = zscore(obj.ts{primary}(tsmi(primary) : tsmi(primary) + obj.sublen - 1), 1);
            hold(ax, 'on');
            for i = 1 : length(tsmi)
                if i == primary
                    plot(ax, zeros(length(S),1));
                    continue;
                end
                plot(ax, S - zscore(obj.ts{i}(tsmi(i) : tsmi(i) + obj.sublen - 1),1));
            end
            hold(ax, 'off');
            drawnow;
        end
        
        function plot_ts(obj,tsi,ax)
            if nargin == 3
                if(~isobject(ax) || ~isvalid(ax))
                    error('invalid graphics object handle');
                end
            else
                fg = figure();
                ax = axes(fg);
            end
            hold(ax,'on');
            for i = 1 : length(tsi)
                if tsi(i) > obj.tsCount
                    error('time series index out of range');
                end
                plot(ax,obj.ts{tsi(i)});
            end
            hold(ax,'off');
            drawnow;
        end
        
        function plot_ss(obj,tsi,tssi,ax)
            if nargin == 3
                fg = figure();
                ax = axes(fg);
            end
            hold(ax,'on');
            for i = 1 : length(tsi)
                if tsi(i) > obj.tsCount
                    error('time series index out of range');
                elseif tssi(i) > length(obj.ts{tsi(i)})
                    error('time series subsequence index is out of range');
                end
                if nargin == 4
                    if isvalid(ax)
                        plot(ax, (obj.ts{tsi(i)}(tssi(i) : tssi(i) + obj.sublen - 1) - obj.mu{tsi(i)}(tssi(i))) .* obj.invn{tsi(i)}(tssi(i)));
                    else
                        error('invalid axis handle');
                    end
                else
                    plot((obj.ts{tsi(i)}(tssi(i) : tssi(i) + obj.sublen - 1) - obj.mu{tsi(i)}(tssi(i))) .* obj.invn{tsi(i)}(tssi(i)));
                end
            end
            hold(ax,'off');
            drawnow;
        end
        
        function data = export_basic(obj)
            % Exports the core data structures used here in the form of a
            % struct. This can be used to export the basic information in cases where 
            % you wish to export a consensus object as a .mat file, and a class definition may not be available later.
            data = struct('ts',   obj.ts, 'sublen', obj.sublen, 'mu', obj.mu,...
                'invn', obj.invn,   'mp', obj.mp,     'mpi',obj.mpi,...
                'ffts', obj.ffts);
        end
        
        function [rad, diam] = comp_rad(obj, nn_indices, seed_index)
            % Utility function to check radius and diameter.
            % Here the radius is the radius of a hypersphere surrounding
            % one subsequence and encompassing all of the others according
            % to nn_indices. Diameter is just the maximum difference
            % between any two.
            if isempty(nn_indices) || isempty(seed_index)
                error('check inputs');
            elseif length(nn_indices) > obj.tsCount
                error('Input index sequence contains more entries than the number of time series available');
            end
            ss = zeros(obj.sublen,length(nn_indices));
            for i = 1 : size(ss,2)
                ss(:,i) = (obj.ts{i}(nn_indices(i) : nn_indices(i) + obj.sublen - 1) - obj.mu{i}(nn_indices(i))) .* obj.invn{i}(nn_indices(i));
            end
            rad = 0;
            for i = 1 : size(ss,2)
                if i == seed_index
                    continue;
                end
                rad = max(rad, norm(zscore(ss(:,seed_index),1) - zscore(ss(:,i),1)));
            end
            diam = rad;
            for i = 1 : size(ss,2)
                if i == seed_index
                    continue;
                end
                z0 = zscore(ss(:,i),1);
                for j = 1 : size(ss,2)
                    if j == i
                        continue;
                    end
                    z1 = zscore(ss(:,j),1);
                    diam = max(diam, norm(z0 - z1));
                end
            end
        end
        
        function [mp,mpi] = ABjoin(obj,a,b)
            %
            % This should support more than one backend. I would prefer to
            % add an mex for faster codepaths
            [mp,mpi] = mpx_AB(obj.ts{a}, obj.ts{b}, obj.sublen);
            
            %mp = zeros(length(obj.ts{a}) - obj.sublen + 1, 1);
            %mpi = zeros(length(obj.ts{a}) - obj.sublen + 1, 1);
            
            %for i = 1 : length(obj.ts{a}) - obj.sublen + 1
            %    q = (obj.ts{a}(i : i + obj.sublen - 1) - obj.mu{a}(i)) .* obj.invn{a}(i);
            %    [mp(i), mpi(i)] = obj.findnearest(q,b);
            %end
        end
    end
    methods(Static)
        function [valid, message] = validateParameters(ts, sublen)
            message = '';
            if ~iscell(ts)
                message = 'Input must be an N x 1 cell array';
            elseif size(ts,1) ~= max(size(ts))
                message = 'Input format is unsupported. It must be an N x 1 cell array';
            end
            m = inf;
            for i = 1 : length(ts)
                if ~isa(ts{i},'double') && ~isa(ts{i},'single')
                    message = 'Only double and single precision are supported. Nested cells, integer arrays, etc are unsupported';
                elseif ~isvector(ts{i})
                    message = 'All time series must be 1D vector types. Multidimensional arrays are not supported';
                end
                m = min(m,length(ts{i}));
            end
            if m < sublen
                message = 'shortest time series is shorter than the desired subsequence length';
            end
            if length(ts) < 2
                message = 'need at least 2 comparator time series';
            end
            for i = 1 : length(ts)
                if size(ts{i},1) < size(ts{i},2)
                    message = 'function accepts a cell array of column based time series';
                end
            end
            valid = isempty(message);
        end
        
        function [sol, obj] = from_nan_cat(ts, sublen, plotsol)
            % Take a time series comprised of multiple concatenated time series
            % This will treat any and all nan values as the end of a time
            % series
            
            if nargin < 2 || isempty(ts)
                error('invalid input');
            end
            f = find(isnan(ts));
            if isempty(f)
                error('requires more than one time series');
            elseif f(1) == 1
                error('input must begin with a valid time series');
            end
            start = zeros(length(f) + 1, 1);
            fin = zeros(length(f) + 1, 1);
            ind = 2;
            start(1) = 1;
            fin(1) = f(1) - 1;
            for i = 1 : length(f) - 1
                if f(i + 1) > f(i) + 1
                    start(ind) = f(i) + 1;
                    fin(ind) = f(i + 1) - 1;
                    ind = ind + 1;
                end
            end
            if f(end) < length(ts)
                start(ind) = f(end) + 1;
                fin(ind) = length(ts);
            end
            ts_c = cell(ind,1);
            for i = 1 : ind
                ts_c{i} = ts(start(i) : fin(i));
            end
            if nargin < 3
                plotsol = false;
            end
            consensus_search.validateParameters(ts_c, sublen);
            obj = consensus_search(ts_c, sublen);
            sol =  obj.solve_opt(plotsol);
        end
        
        function m = min_len(ts)
            m = inf;
            for i = 1:length(ts)
                m = min(m, length(ts{i}));
            end
        end

        function [mx, mxi] = mink(A, k)
        %mink - Find k smallest elements of array
        %
        % Syntax: mx = mink(A, k)
            [srt, idx] = sort(A);
            mx = srt(1:k);
            mxi = idx(1:k);
        end
    end
end