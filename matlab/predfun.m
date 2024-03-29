function yfit = predfun(Xtrain,ytrain,Xtest)

regressionGP = fitrgp( ...
    Xtrain, ...
    ytrain, ...
    'BasisFunction', 'constant', ...
    'KernelFunction', 'exponential', ...
    'Standardize', true);
yfit = predict(regressionGP,Xtest);
end