

clear all;

[signal, samplerate] = audioread('1.wav');
[mask, samplerate2] = audioread('1_mask.wav');

assert(samplerate == samplerate2)

sampleperiod = 1 / samplerate;

sigt = (0:length(signal)-1).*sampleperiod;
maskt = (0:length(mask)-1).*sampleperiod;

#figure;
#plot(sigt, signal);
#error('123');


myfile = './tmp/my_fbank.bin';
pyfile = './tmp/py_fbank.bin';
my_sgram_file = './tmp/my_sgram.bin';
py_sgram_file = './tmp/py_sgram.bin';
dtype = 'float64'
nfilts = 10

fft_size = 256
env_size = fft_size/2 + 1

envperiod = fft_size / samplerate

F = fopen(myfile, 'rb');
myraw = fread(F, Inf, dtype);
fclose(F);

F = fopen(pyfile, 'rb');
pyraw = fread(F, Inf, dtype);
fclose(F);

assert(length(myraw) == length(pyraw));

nframes = length(myraw) / nfilts;

envt = (0 : nframes - 1) * envperiod;

myenvs = reshape(myraw, nfilts, nframes)';
pyenvs = reshape(pyraw, nfilts, nframes)';

F = fopen(my_sgram_file, 'rb');
mysgramraw = fread(F, Inf, dtype);
fclose(F);

F = fopen(py_sgram_file, 'rb');
pysgramraw = fread(F, Inf, dtype);
fclose(F);

mysgram = reshape(mysgramraw, env_size, nframes)';
pysgram = reshape(pysgramraw, env_size, nframes)';

myenvs_log = 10*log10(myenvs);
pyenvs_log = 10*log10(pyenvs);
#
#figure;
#surf(myenvs);

#figure;
#surf(pyenvs);

N = 3

figure;
hold on;
plot( envt, (myenvs_log(:,N)) );
plot( envt, (pyenvs_log(:,N)), 'r');
plot( sigt, signal .* 60, 'g');
plot( maskt, mask .* 100, 'k');
legend('my', 'py', 'signal', 'mask')

#figure;
#hold on;
#plot( diff(myenvs_log(:,N)) )
#plot( diff(pyenvs_log(:,N)), 'r')
#legend('my', 'py')

#figure;
#surf(mysgram);
#title('My sgram')
#figure;
#surf(pysgram)
#title('PY sgram')


#dmean = mean(diff(myenvs)(:,1:5),2);
#figure;
#plot(dmean);



