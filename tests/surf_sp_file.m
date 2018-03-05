
clear all;

dtype = 'float64';

inp_file = argv(){1};
sp_dim = str2num(argv(){2});

fprintf('Input file : %s    sp-dim : %f \n', inp_file, sp_dim);

fid = fopen(inp_file);
rawdata = fread(inp_file, Inf, dtype);
fclose(fid);

if ( rem(length(rawdata), sp_dim) != 0)
    fprintf('ERROR: unexpected input size of sp_dim (%d  %d  %f) \n', length(rawdata), sp_dim, ...
        length(rawdata) / sp_dim);
end

numframes = length(rawdata) / sp_dim;
data = reshape(rawdata, sp_dim, numframes);

figure;
surf(data);
input('Press any key ... ');



