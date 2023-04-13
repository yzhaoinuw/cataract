function [error, std_error, max_error,error_for_ANOVA_SRKT_vector] = test_error_SRKT_function(Feature_Matrix)

% load Features5
% % % % % EXCEL
% % % % % Column A (1): Sex: 1M2F
% % % % % Column B (2): Laterality: 1OD2OS
% % % % % Column C (3): AgeAtTimeOfOperationyear
% % % % % Column D (4): IOLModel: 1SN60WF2CNA0T03MX60EUS
% % % % % Column E (5): IOLPowerInsertedD 
% % % % % Column F (6): AxialLengthmm IOLMaster
% % % % % Column G (7): PreopK1
% % % % % Column H (8): PreopK1Axis 
% % % % % Column I (9): PreopK2 
% % % % % Column J (10): PreopK2Axis 
% % % % % Column K (11): Sphere 
% % % % % Column L (12): Cyl
% % % % % Column M (13): SphericalEquiv
% % % % % Column N (14): Number of days to post-op scan
% % % % % Column O (15): Pupil size
% % % % % Column P (16): Radius of curvature of Anterior Cornea (RAC)
% % % % % 
% % % % % PRE_OCT
% % % % % (17): Corneal Thickness (CT)
% % % % % (18): ACD
% % % % % (19): LT
% % % % % (20): Vitreou chamber depth (VCD)
% % % % % (21): Axial Length (AL) 
% % % % % (22): AL_not_corrected
% % % % % (23): std_AL_non_corrected_eyes
% % % % % (24): med_RAC_eyes
% % % % % (25): med_RPC_eyes
% % % % % (26): med_RAL_eyes
% % % % % (27): med_RPL_eyes
% % % % % (28): med_RAC_eyes_Diam2
% % % % % (29): med_RPC_eyes_Diam2
% % % % % (30): med_RAL_eyes_Diam2
% % % % % (31): med_RPL_eyes_Diam2
% % % % % (32): RAC_3D
% % % % % (33): RPC_3D
% % % % % (34): RAL_3D
% % % % % (35): RPL_3D
% % % % % (36): RAC_3D_Diam2
% % % % % (37): RPC_3D_Diam2
% % % % % (38): RAL_3D_Diam2
% % % % % (39): RPL_3D_Diam2
% % % % % 
% % % % % 
% % % % % POST_OCT
% % % % % (40): CT_post_eyes
% % % % % (41): IOLT_eyes
% % % % % (42): VCD_post_eyes
% % % % % (43): AL_post_eyes
% % % % % (44): AL_non_corrected_post_eyes
% % % % % (45): ELP_eyes
% % % % % 
% % % % % POST_EXCEL
% % % % % (46): AL_post IOLMaster
% % % % % (47): Sphere post
% % % % % (48) Cylinder post
% % % % % (49) Spherical equiv post

IOL_model=Feature_Matrix(:,4);
Model1=find(IOL_model==1);
Model2=find(IOL_model==2);
Model3=find(IOL_model==3);
A_cons=zeros(size(IOL_model));
A_cons(Model1)=119;
A_cons(Model2)=119.1;
A_cons(Model3)=119.1;

RAC=Feature_Matrix(:,16);
AL_IOLMaster=Feature_Matrix(:,6);
IOLT=Feature_Matrix(:,41);

% ELP is defined as the distance between posterior cornea and anterior IOL 
ELP=Feature_Matrix(:,45);
ELP=ELP+IOLT/2;

ACD_post=[];
for i=1:length(A_cons)
A_cons_eye= A_cons(i);
RAC_eye=RAC(i);
AL_IOLMaster_eye=AL_IOLMaster(i);
ACD_post(i)=SRKT_ACDpos_calculation(A_cons_eye,RAC_eye,AL_IOLMaster_eye);
end
% ACD post in SRKT is defined from posterior cornea to IOLThickness/2 (see
% paper "Development of the SRK/T intraocular lens implant power calculation
% formula")
ACD_post=ACD_post';

error_SRKT=mean(abs(ELP-ACD_post))

ACD_post_no_bias=ACD_post+mean(ELP-ACD_post);
% figure,
% plot(ELP), hold on
% plot(ACD_post_no_bias)
error=mean(abs(ELP-ACD_post_no_bias))
std_error=std(abs(ELP-ACD_post_no_bias))
max_error=max(abs(ELP-ACD_post_no_bias))
error_for_ANOVA_SRKT_vector=abs(ELP-ACD_post_no_bias);
% 
% remove_bias=ELP-ACD_post-mean(ELP-ACD_post)
% error_SRKT2=mean(abs(remove_bias))
% error_mean=mean(abs(ELP-mean(ELP)))
% 
% plot(abs(remove_bias))
% hold on
% plot(abs(ELP-mean(ELP)))


