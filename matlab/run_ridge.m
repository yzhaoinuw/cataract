clear all
close all

load Features8_pi6

Num_exp = 500;
mae = zeros(1,Num_exp);
for i=1:Num_exp

    % RAC, new AL, IOL Model, Axial Measurements
    data = [
        ... % LP
        Feature_Matrix(:,45), ...
        ... # sex
        dummyvar(categorical(Feature_Matrix(:,1))), ...
        ... #laterality
        dummyvar(categorical(Feature_Matrix(:,2))), ...
        ... # IOL model
        dummyvar(categorical(Feature_Matrix(:,4))), ... 
        Feature_Matrix(:,3), ...
        Feature_Matrix(:, [5:44, 46:end]), ...
        ];
    
    % Separate to training and test data
    nanRows = any(isnan(data), 2);
    data = data(~nanRows, :);
    cv = cvpartition(size(data, 1),'HoldOut',0.2);
    idx = cv.test;
    
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);
    
    X_train=[dataTrain(:,2:end)];
    X_test=[dataTest(:,2:end)];
    
    y_train=dataTrain(:,1);
    y_test=dataTest(:,1);
    
    b = ridge(y_train, X_train, 1, 0);
    y_pred = b(1) + X_test * b(2:end);
    % y_pred = predict(model,X_test);
    mae(i) = mean(abs(y_pred-y_test));
end

mean(mae)
std(mae)