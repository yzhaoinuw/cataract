clear all
close all

load Features8_pi7

nExp = 200;
nFolds = 5;
mae = zeros(1, nExp);
X = [...
    ... # sex
    dummyvar(categorical(Feature_Matrix(:, 1))), ...
    ... #laterality
    dummyvar(categorical(Feature_Matrix(:, 2))), ...
    ... # IOL model
    dummyvar(categorical(Feature_Matrix(:, 4))), ... 
    Feature_Matrix(:, 3), ...
    Feature_Matrix(:, [5:44, 46:end]), ...
];

 y = [... % LP
        Feature_Matrix(:, 45), ...
        ];

% drop nan rows
nanRows = any(isnan(X), 2);
X = X(~nanRows, :);
y = y(~nanRows, :); 

indRemap = zeros(1, size(X, 2));
indRemap(1:2) = 1;
indRemap(3:4) = 2;
indRemap(5:7) = 4;
indRemap(8) = 3;
indRemap(9:48) = 5:44;
indRemap(49:end) = 46:size(Feature_Matrix, 2);

featureImportance = zeros(1, size(X, 2));

for i = 1:nExp
    cv = cvpartition(size(X, 1), 'KFold', nFolds);
    maeFold = zeros(1, nFolds);
    for j = 1:nFolds
        % Get training and test sets for this fold
        X_train = X(cv.training(j), :);
        y_train = y(cv.training(j));
        X_test = X(cv.test(j), :);
        y_test = y(cv.test(j));
        
        % Perform ridge regression on the training set
        b = ridge(y_train, X_train, 1, 0);
        weights = abs(b(2:end));
        featureImportance = featureImportance + weights';
        y_pred = b(1) + X_test * b(2:end);
        maeFold(j) = mean(abs(y_pred - y_test));
    end
    mae(i) = mean(maeFold);
end

mean(mae)
std(mae)
[sortedWeights, indices] = sort(featureImportance, 'descend');
indRemap(indices(1:10))