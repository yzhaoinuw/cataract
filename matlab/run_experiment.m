clear all
close all

load Features7

Num_exp=1000;
mae = zeros(1,Num_exp);
for i=1:Num_exp

    % RAC, new AL, IOL Model, Axial Measurements
    data = [
        ... %LP
        Feature_Matrix(:,45), ...
        ... # IOL model
        dummyvar(categorical(Feature_Matrix(:,4))), ... 
        ... % old AL
        ...Feature_Matrix(:,6), ...
        ... % RAC
        Feature_Matrix(:,16), ... 
        ... % Axial Measurements
        Feature_Matrix(:,17), ...
        Feature_Matrix(:,18), ...
        Feature_Matrix(:,19), ...
        Feature_Matrix(:,20), ...
        ... % new AL
        Feature_Matrix(:,21), ... 
        ... % crystaline features
        Feature_Matrix(:,34), ...
        Feature_Matrix(:,35), ...
        ];
    
    % Separate to training and test data
    data(any(isnan(data), 2), :) = [];
    cv = cvpartition(size(data,1),'HoldOut',0.2);
    idx = cv.test;
    
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);
    
    X_train=[dataTrain(:,2:end)];
    X_test=[dataTest(:,2:end)];
    
    y_train=dataTrain(:,1);
    y_test=dataTest(:,1);
    
    
    model = fitrgp( ...
        X_train, ...
        y_train, ...
        'BasisFunction', 'constant', ...
        'KernelFunction', 'exponential', ...
        'Standardize', true);
    y_pred = predict(model,X_test);
    mae(i) = mean(abs(y_pred-y_test));
end

mean(mae)
std(mae)