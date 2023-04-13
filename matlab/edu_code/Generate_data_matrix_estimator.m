%%%%%%%%%%%%%%%%%
%   ARVO 2023
%%%%%%%%%%%%%%%%%

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


% X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,17:21) Feature_Matrix(:,26:27) ];
X=[Feature_Matrix(:,16) Feature_Matrix(:,4) Feature_Matrix(:,17:21)];
% X=Feature_Matrix; 
Y=Feature_Matrix(:,45);