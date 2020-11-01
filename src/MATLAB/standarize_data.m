function res = standarize_data(data)

res = (data - mean(data))/std(data);

end