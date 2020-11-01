function output_signal = butterworth2(input_singal,fc,fs)
[b, a] = butter(2, fc/(fs/2), 'low');
output_signal = filtfilt(b, a, input_singal);
end