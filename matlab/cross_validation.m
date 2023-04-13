clear all
close all

load Features7


%%%%%%%%%%%%%%%%%%%% RAC, IOLModel, CT, ACD, LT, VCD, AL
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
    %Feature_Matrix(:,34), ...
    %Feature_Matrix(:,35), ...
    ];

data(any(isnan(data), 2), :) = [];
X=[data(:,2:end)];
y=data(:,1);

K = 5;
Num_exp=200;
error = zeros(1,Num_exp);
error_std=zeros(1,Num_exp);
for j=1:Num_exp
    cv = cvpartition(size(X,1),'KFold',K);
    mae_cv = zeros(K,1);
    std_cv = zeros(K,1);
    for i = 1:K
        % Get the training and testing data for this fold
        Xtrain = X(cv.training(i),:);
        ytrain = y(cv.training(i));
        Xtest = X(cv.test(i),:);
        ytest = y(cv.test(i));
        
        % Fit a model using the training data
        model = fitrgp( ...
            Xtrain, ...
            ytrain, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', 'exponential', ...
            'Standardize', true);
        
        % Predict the target variable using the test data
        ypred = predict(model,Xtest);
        
        % Compute the MAE for this fold
        mae_cv(i) = mean(abs(ytest - ypred));
        std_cv(i) = std(abs(ytest - ypred));
    end
    error(j)=mean(mae_cv);
    error_std(j)=mean(std_cv);
end
error;
mean(error)
mean(error_std)
    