%%%%%%%%%%%%%%%%%
%   ARVO 2023
%%%%%%%%%%%%%%%%%
% 10 fold GRP Exponential.


clear all
close all


load Features5
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


limit_value=0.2;
%%%%%%%%%%%%%%%%%%%% RAC, IOLModel, CT, ACD, LT, VCD, AL(OCT), RAL3D, RPL3D
%{
X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,17:21) Feature_Matrix(:,34:35)];
Y=Feature_Matrix(:,45);

Num_exp=10;
Error=[];
Std_Error=[];
max_Error=[];
higher_limit_value=[];
error_for_ANOVA_FD=[];

for i=1:Num_exp

    [trainedModel, validationRMSE,validationPredictions] = Regression_GPR_9var(X, Y);
    MAE=mean(abs(Y-validationPredictions));
    std_MAE=std(abs(Y-validationPredictions));
    Error=[Error MAE];
    Std_Error=[Std_Error std_MAE];
    max_Error_exp=max(abs(Y-validationPredictions));
    max_Error=[max_Error max_Error_exp];
    higher_limit_value=[higher_limit_value length(find(abs(Y-validationPredictions)>limit_value))];
%     validationPredictions_set_of_feat_D=[validationPredictions_set_of_feat_D validationPredictions]; % Predictions of the model in validation set
    error_for_ANOVA_FD=[error_for_ANOVA_FD abs(Y-validationPredictions)];
end

Mean_MAE=mean(Error)
STD_exper=std(Error) % STD across experiments
STD_subj=mean(Std_Error) % STD across subjects (mean across experiments)
Mean_max=mean(max_Error) % Max error across subjects (mean across experiments)
num_higher_limit_value=mean(higher_limit_value) % only for the last experiment
error_for_ANOVA_FD_vector=mean(error_for_ANOVA_FD,2);
%}
%%%%%%%%%%%%%%%%%%%% RAC, IOLModel, CT, ACD, LT, VCD, AL
X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,17:21)];
Y=Feature_Matrix(:,45);

Num_exp=100;
Error_7var=[];
Std_Error=[];
max_Error=[];
higher_limit_value=[];
error_for_ANOVA_FC=[];

for i=1:Num_exp
    [trainedModel, validationRMSE,validationPredictions] = Regression_GPR_7var(X, Y);
    MAE=mean(abs(Y-validationPredictions));
    std_MAE=std(abs(Y-validationPredictions));
    Error_7var=[Error_7var MAE];
    Std_Error=[Std_Error std_MAE];
    max_Error_exp=max(abs(Y-validationPredictions));
    max_Error=[max_Error max_Error_exp];
    higher_limit_value=[higher_limit_value length(find(abs(Y-validationPredictions)>limit_value))];
    error_for_ANOVA_FC=[error_for_ANOVA_FC abs(Y-validationPredictions)];
end
Mean_MAE=mean(Error_7var)
STD_exper=std(Error_7var) % STD across experiments
STD_subj=mean(Std_Error) % STD across subjects (mean across experiments)
Mean_max=mean(max_Error) % Max error across subjects (mean across experiments)
num_higher_limit_value=length(find(abs(Y-validationPredictions)>limit_value)) 
error_for_ANOVA_FC_vector=mean(error_for_ANOVA_FC,2);

%%%%%%%%%%%%%%%%%%%% RAC, IOLModel, AL (OCT)
%{
X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,21)];
Y=Feature_Matrix(:,45);

Num_exp=10;
Error_3var=[];
Std_Error=[];
max_Error=[];
higher_limit_value=[];
error_for_ANOVA_FB=[];

for i=1:Num_exp
    [trainedModel, validationRMSE,validationPredictions] = Regression_GPR_3var(X, Y);
    MAE=mean(abs(Y-validationPredictions));
    std_MAE=std(abs(Y-validationPredictions));
    Error_3var=[Error_3var MAE];
    Std_Error=[Std_Error std_MAE];
    max_Error_exp=max(abs(Y-validationPredictions));
    max_Error=[max_Error max_Error_exp];
    higher_limit_value=[higher_limit_value length(find(abs(Y-validationPredictions)>limit_value))];
    error_for_ANOVA_FB=[error_for_ANOVA_FB abs(Y-validationPredictions)];
end
Mean_MAE=mean(Error_3var)
STD_exper=std(Error_3var) % STD across experiments
STD_subj=mean(Std_Error) % STD across subjects (mean across experiments)
Mean_max=mean(max_Error) % Max error across subjects (mean across experiments)
num_higher_limit_value=length(find(abs(Y-validationPredictions)>limit_value))
error_for_ANOVA_FB_vector=mean(error_for_ANOVA_FB,2);

%%%%%%%%%%%%%%%%%%%% RAC, IOLModel, AL (IOLMaster)
X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,6)];
Y=Feature_Matrix(:,45);

Num_exp=10;
Error_3var=[];
Std_Error=[];
max_Error=[];
higher_limit_value=[];
error_for_ANOVA_FA=[];

for i=1:Num_exp
    [trainedModel, validationRMSE,validationPredictions] = Regression_GPR_3var(X, Y);
    MAE=mean(abs(Y-validationPredictions));
    std_MAE=std(abs(Y-validationPredictions));
    Error_3var=[Error_3var MAE];
    Std_Error=[Std_Error std_MAE];
    max_Error_exp=max(abs(Y-validationPredictions));
    max_Error=[max_Error max_Error_exp];
    higher_limit_value=[higher_limit_value length(find(abs(Y-validationPredictions)>limit_value))];    
    error_for_ANOVA_FA=[error_for_ANOVA_FA abs(Y-validationPredictions)];
end
Mean_MAE=mean(Error_3var)
STD_exper=std(Error_3var) % STD across experiments
STD_subj=mean(Std_Error) % STD across subjects (mean across experiments)
Mean_max=mean(max_Error) % Max error across subjects (mean across experiments)
num_higher_limit_value=length(find(abs(Y-validationPredictions)>limit_value))
error_for_ANOVA_FA_vector=mean(error_for_ANOVA_FA,2);
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SRK/T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[error_SRKT, std_error_SRKT, max_error_SRKT,error_for_ANOVA_SRKT_vector] = test_error_SRKT_function(Feature_Matrix)

