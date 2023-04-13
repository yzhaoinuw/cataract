function [ACD] = SRKT_ACDpos_calculation(A_cons,r,L )

% V=12; na=1.336; nc=1.333; ncml=0.333;

ACDcons=0.62467*A_cons-68.747;
L_COR=L;
if L>24.2
    L_COR=-3.446+1.715*L-0.0237*L^2;
end
K=337.5/r;
Cw=-5.41+0.58412*L_COR+0.098*K;

H=r-sqrt(r^2-(Cw^2)/4);

offset=ACDcons-3.336;
ACD=H+offset;    


% RETHICK=0.65696-0.02029*L;
% LOPT=L+RETHICK;
% 
% IOLemme= (1000*na*(na*r-ncml*LOPT))/((LOPT-ACD)*(na*r-ncml*ACD))
% 
% 
% IOL=23;
% REFX=(1000*na*(na*r-ncml*LOPT)-IOL*(LOPT-ACD)*(na*r-ncml*ACD))/(na*(V*(na*r-ncml*LOPT)+LOPT*r)-0.001*IOL*(LOPT-ACD)*(V*(na*r-ncml*ACD)+ACD*r))



end

