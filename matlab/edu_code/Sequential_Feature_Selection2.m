%%%%%%%%%%%%%%%%%
%   ARVO 2023
%%%%%%%%%%%%%%%%%
% 10 fold GRP Exponential.


clear all
close all

% ARVO 2023 uses Features5, 41 data
% load Features5
addpath('.\Features')

% Features6, including 16 new data (January)
% load Features6 % Total 55 data
% Features7, including new data (February)
% load Features8_pi6 % Total 61 data
load Features8_pi7

% EXCEL
% Column A (1): Sex: 1M2F
% Column B (2): Laterality: 1OD2OS
% Column C (3): AgeAtTimeOfOperationyear
% Column D (4): IOLModel: 1SN60WF2CNA0T03MX60EUS
% Column E (5): IOLPowerInsertedD
% Column F (6): AxialLengthmm IOLMaster
% Column G (7): PreopK1
% Column H (8): PreopK1Axis
% Column I (9): PreopK2
% Column J (10): PreopK2Axis
% Column K (11): Sphere
% Column L (12): Cyl
% Column M (13): SphericalEquiv
% Column N (14): Number of days to post-op scan
% Column O (15): Pupil size
% Column P (16): Radius of curvature of Anterior Cornea (RAC)
%
% PRE_OCT
% (17): Corneal Thickness (CT)
% (18): ACD
% (19): LT
% (20): Vitreou chamber depth (VCD)
% (21): Axial Length (AL)
% (22): AL_not_corrected
% (23): std_AL_non_corrected_eyes
% (24): med_RAC_eyes (2 mm en Features5_b; 3 mm en Features5)
% (25): med_RPC_eyes
% (26): med_RAL_eyes
% (27): med_RPL_eyes
% (28): med_RAC_eyes_Diam2 (6mm)
% (29): med_RPC_eyes_Diam2
% (30): med_RAL_eyes_Diam2
% (31): med_RPL_eyes_Diam2
% (32): RAC_3D
% (33): RPC_3D
% (34): RAL_3D
% (35): RPL_3D
% (36): RAC_3D_Diam2
% (37): RPC_3D_Diam2
% (38): RAL_3D_Diam2
% (39): RPL_3D_Diam2
%
%
% POST_OCT
% (40): CT_post_eyes
% (41): IOLT_eyes
% (42): VCD_post_eyes
% (43): AL_post_eyes
% (44): AL_non_corrected_post_eyes
% (45): ELP_eyes
%
% POST_EXCEL
% (46): AL_post IOLMaster
% (47): Sphere post
% (48) Cylinder post
% (49) Spherical equiv post
%
% PRE_OCT (FULL SHAPE)
% (50): VOL_eigen_lenses
% (51): LSA_eigen_lenses
% (52): DIA_eigen_lenses
% (53): EPP_eigen_lenses
% (54): LT_eigen_lenses
% (55): Coef_eigenlenses, a1
% (56): Coef_eigenlenses, a2
% (57): Coef_eigenlenses, a3
% (58): Coef_eigenlenses, a4
% (59): Coef_eigenlenses, a5
% (60): Coef_eigenlenses, a6
% (61): Coef_eigencenters, c1
% (62): Coef_eigencenters, c2
% (63): Coef_eigencenters, c3
% (64): Coef_eigencenters, c4
% (65): Coef_eigencenters, c5
% (66): Coef_eigencenters, c6
% (67): EPP_eigen_lenses2
% (68): LT_IOVS
% (69): VOL_IOVS
% (70): LSA_IOVS
% (71): DIA_IOVS
% (72): EPP_IOVS

% Filter the set of Features that we want to evaluate (Always exclude the
% label! Feature_Matrix(:,45))
X=[Feature_Matrix(:,1:39) Feature_Matrix(:,50:end)];
% X=[Feature_Matrix(:,50:end)];
Y=Feature_Matrix(:,45);

% Parameters for the sequentialfs algorithm
N_exp=200;
K_number_of_folds=5; % 5-fold--> 20 % for testing. Reasonble (better than just the 10 %)
c = cvpartition(Y,'k',K_number_of_folds,'Stratify',false);
opts = statset('Display','iter','TolFun',1e-8,'TolTypeFun', 'abs');
% opts = statset('Display','iter');
% 'direction': The direction of the sequential search. The default is 'forward'. A value of 'backward' specifies an initial candidate set including all features and an algorithm that removes features sequentially until the criterion increases.
% 'nfeatures': The number of features at which sequentialfs should stop. inmodel includes exactly this many features. The default value is empty, indicating that sequentialfs should stop when a local minimum of the criterion is found. A nonempty value overrides values of 'MaxIter' and 'TolFun' in 'options'.
% 'keepin': A logical vector or a vector of column numbers specifying features that must be included. The default is empty
% 'keepout': A logical vector or a vector of column numbers specifying features that must be excluded. The default is empty
% Functin handle
% fun = @(train_data,train_labels,test_data,test_labels)loss(fitrgp(...
%     train_data, ...
%     train_labels, ...
%     'BasisFunction', 'constant', ...
%     'KernelFunction', 'exponential', ...
%     'Standardize', true),test_data,test_labels);


% classf = @(train_data,train_labels,test_data,test_labels) ...
%        sum(predict(fitcsvm(train_data,train_labels,'KernelFunction','rbf'), test_data) ~= test_labels); 


% for i=1:10
% % stream = RandStream('mt19937ar','Seed',0);
% % reset(stream);
% % cnew = repartition(c,stream,'Stratify',false);
% cnew = cvpartition(Y,'k',K_number_of_folds,'Stratify',false);
% fold1 = test(cnew,2)
% find(fold1==1)
% end


imodel_matrix=[];
crit_matrix={};
for Num_exp=1:N_exp
% rng('shuffle')
% cnew = repartition(c);
% fold1 = test(cnew,1)
% find(fold1==1)
% OJO a esto de la estratificación en el sampling! 
% También pensar en la normalización que haremos en future test samples
c = cvpartition(Y,'k',K_number_of_folds,'Stratify',false);
fun=@(train_data,train_labels,test_data,test_labels)loss_MAE_RGP(train_data,train_labels,test_data,test_labels);
[inmodel,history] = sequentialfs(fun,X,Y,'cv',c,'options',opts) 

imodel_matrix=[imodel_matrix;inmodel]
crit_matrix{Num_exp}=history.Crit; 

end

% If the frature selected is above 45, be carefull because the index
% references a Feature in the index+1 (becasue the label, column 45, was
% removed)

a=sum(imodel_matrix);
[ind times]=sort(a,'descend');
% OJO!. Así no coge la mejor combinación, puede coger características
% redundantes... (por ej, AL_IOLMaster y AL_noncorrected) que no hubieran
% sido elegidas en combinación...
% Hay que meter el buccle de num de experimentos dentro de la función de
% coste!!!
% Ver comentarios en el móvil!!!
% Pensar qué puede hacr Yue
