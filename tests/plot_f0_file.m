
clear all;

dtype = 'float64';

inp_file = argv(){1};

fprintf('Input file : %s \n', inp_file);

fid = fopen(inp_file);
rawdata = fread(inp_file, Inf, dtype);
fclose(fid);

figure;
plot(rawdata);
input('Press any key ... ');


