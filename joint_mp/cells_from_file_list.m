
function ts = cells_from_file_list(ts_dir, rowsorcols, num)

% This is a convenience function, but it's not that highly
% tested on different datasets. It assumes a file list is
% generated using matlab's dir function with whatever filtering
% is required.

if strcmp(rowsorcols,'rows')
    [ts] = load_from_files(ts_dir,'rows', num);
elseif strcmp(rowsorcols,'cols')
    [ts] = load_from_files(ts_dir,'cols', num);
else
    error('check input');
end

end

function ts = load_from_files(f, format, num)
ts = cell(length(f), 1);
if strcmp('rows',format)
    for i = 1 : length(f)
        ts{i} = load(strcat(f(i).folder,filesep,f(i).name));
        if size(ts{i},1) ~= 1
            ts{i} = ts{i}(num, :);
        end
    end
elseif strcmp('cols',format)
    for i = 1 : length(f)
        ts{i} = load(strcat(f(i).folder,filesep,f(i).name));
        if size(ts{i}, 2) ~= 1
            ts{i} = ts{i}(:, num);
        end
    end
else
    error('format unsupported, choose rows or cols');
end
end
