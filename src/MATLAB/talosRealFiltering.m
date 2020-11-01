%% Samping Rate
fs = 25;
start = 500;
endl = 3100;
%% Filter the Angular Velocity of Leg IMUs
for i = 1 : dlen
   lgyroB(i,:) = (Rotwb{i}' * lgyroW(i,:)')';
   rgyroB(i,:) = (Rotwb{i}' * rgyroW(i,:)')';
end
fc = 3;
lgyroBf = butterworth2(lgyroB,fc,fs);
rgyroBf  = butterworth2(rgyroB,fc,fs);
%% Filter the Angular Velocity Measurements.
fc = 4;
lwf = butterworth2(lw,fc,fs);
rwf  = butterworth2(rw,fc,fs);
fc = 4;
gyrof = butterworth2(gyro,fc,fs);
gyrodotf = diff([0,0,0;gyrof]*fs);

lwf_est = lgyroBf - gyrof;
rwf_est = rgyroBf - gyrof;
figure
plot(lwf(start:endl,3),'black');
hold on
plot(lwf_est(start:endl,3),'blue');


figure
plot(gyrof(start:endl,1),'black','linewidth',1);
hold on
plot(-rwf_est(start:endl,1),'red');
hold on
plot(-lwf_est(start:endl,1),'green');
%% Filter the Linear Acceleration Measurements
fc = 3.5;
accf = butterworth2(acc,fc,fs);
%accd = delayseq(accf,1);
accd = accf;
% figure
% plot(acc(start:endl,3))
% hold on
% plot(accd(start:endl,3))



%%Linear Velocities
%Left-Leg
fc = 3;
lvf = butterworth2(lv,fc,fs);
laf = [0 0 0;diff(lvf)*fs];
%Right-Leg
rvf = butterworth2(rv,fc,fs);
raf = [0 0 0;diff(rvf)*fs];
lposf = butterworth2(lpos,fc,fs);
rposf = butterworth2(rpos,fc,fs);
% acc_LLeg = -laf - cross(gyrodotf,lpos) - cross(gyrod,lv); 
% acc_RLeg = -raf - cross(gyrodotf,rpos) - cross(gyrod,rv); 
% fc = 3;
% acc_LLeg= butterworth2(acc_LLeg,fc,fs);
% acc_RLeg= butterworth2(acc_RLeg,fc,fs);
% 
for i = 1 : dlen
   laccB(i,:) = (Rotwb{i}' * laccW(i,:)')';
   raccB(i,:) = (Rotwb{i}' * raccW(i,:)')';
end
fc = 3.0;
laccBf = butterworth2(laccB,fc,fs);
fc = 3.0;
raccBf  = butterworth2(raccB,fc,fs);
% figure
% plot(raccBf(start:endl,3))
% hold on
% plot(raccB(start:endl,3))


laf_est = laccBf - accf - cross(gyrodotf,lpos) - cross(gyrof,lv); 
raf_est = raccBf - accf - cross(gyrodotf,rpos) - cross(gyrof,lv); 
acc_LLeg_est = -laf_est - cross(gyrodotf,lpos) - cross(gyrof,lv); 
acc_RLeg_est = -raf_est - cross(gyrodotf,rpos) - cross(gyrof,rv); 

% acc_RLeg_est  = butterworth2(acc_RLeg_est,fc,fs);
% acc_LLeg_est  = butterworth2(acc_LLeg_est,fc,fs);
% 
%acc_RLeg = accWf + (cross(gyrodotWf,rposW) + cross(gyroWf,cross(gyroWf,rposW)) + 2.0*cross(gyroWf,rvWf) + raWf);
%acc_LLeg = accWf + (cross(gyrodotWf,lposW) + cross(gyroWf,cross(gyroWf,lposW)) + 2.0*cross(gyroWf,lvWf) + laWf);
figure
plot(accf(start:endl,3),'black');
hold on
plot(acc_LLeg_est(start:endl,3),'green');
hold on
plot(acc_RLeg_est(start:endl,3),'red');

if(saveData == 1)
    %LLeg Label
    dlmwrite(strcat(saveDir,'/baccX_LL.txt'),acc_LLeg_est(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_LL.txt'),acc_LLeg_est(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_LL.txt'),acc_LLeg_est(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/baccX_RL.txt'),acc_RLeg_est(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_RL.txt'),acc_RLeg_est(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_RL.txt'),acc_RLeg_est(1:dlen,3))
    dlmwrite(strcat(saveDir,'/baccX.txt'),accf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY.txt'),accf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ.txt'),accf(1:dlen,3))
    
    dlmwrite(strcat(saveDir,'/bgX_LL.txt'),-lwf_est(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY_LL.txt'),-lwf_est(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ_LL.txt'),-lwf_est(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/bgX_RL.txt'),-rwf_est(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY_RL.txt'),-rwf_est(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ_RL.txt'),-rwf_est(1:dlen,3))
    dlmwrite(strcat(saveDir,'/bgX.txt'),gyrof(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY.txt'),gyrof(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ.txt'),gyrof(1:dlen,3))
end
