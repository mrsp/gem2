%% Samping Rate
fs = 100;

%% Filter the Angular Velocity of Leg IMUs
for i = 1 : dlen
   lgyroB(i,:) = (Rotwb{i}' * lgyroW(i,:)')';
   rgyroB(i,:) = (Rotwb{i}' * rgyroW(i,:)')';
end
fc = 10;
lgyroBf = butterworth2(lgyroB,fc,fs);
fc = 10;
rgyroBf  = butterworth2(rgyroB,fc,fs);
% figure
% plot(laccBf(:,2))
% hold on
% plot(raccBf(:,2))
% 




%% Filter the Angular Velocity Measurements.
fc = 5;
lwf = butterworth2(lw,fc,fs);
rwf  = butterworth2(rw,fc,fs);
fc = 10;
gyrof = butterworth2(gyro,fc,fs);
figure
plot(gyro(:,2),'red','linewidth',1);
hold on
plot(gyrod(:,2),'blue','linewidth',1);
gyrodotf = diff([0,0,0;gyrof]*fs);
lwf_est = lgyroBf - gyrof;
rwf_est = rgyroBf - gyrof;
% figure
% plot(lwf(:,1),'black');
% hold on
% plot(lwf_est(:,1),'blue');


figure
plot(gyrof(:,1),'black','linewidth',1);
hold on
plot(-rwf_est(:,1),'red');
hold on
plot(-lwf_est(:,1),'green');
%% Filter the Linear Acceleration Measurements
fc = 7.0;
accf = butterworth2(acc,fc,fs);
%accd = delayseq(accf,1);
accd = accf;
% figure
% plot(acc(:,3))
% hold on
% plot(accd(:,3))



%%Linear Velocities
%Left-Leg
fc = 5;
lvf = butterworth2(lv,fc,fs);
laf = [0 0 0;diff(lvf)*fs];
%Right-Leg
rvf = butterworth2(rv,fc,fs);
raf = [0 0 0;diff(rvf)*fs];
lposf = butterworth2(lpos,fc,fs);
rposf = butterworth2(rpos,fc,fs);

% 
for i = 1 : dlen
   laccB(i,:) = (Rotwb{i}' * laccW(i,:)')';
   raccB(i,:) = (Rotwb{i}' * raccW(i,:)')';
end
fc = 7.0;
laccBf = butterworth2(laccB,fc,fs);
raccBf  = butterworth2(raccB,fc,fs);
% figure
% plot(raccBf(:,3))
% hold on
% plot(raccB(:,3))


laf_est = laccBf - accf - cross(gyrodotf,lpos) - cross(gyrod,lv); 
raf_est = raccBf - accf - cross(gyrodotf,rpos) - cross(gyrod,lv); 
acc_LLeg_est = -laf_est - cross(gyrodotf,lpos) - cross(gyrod,lv); 
acc_RLeg_est = -raf_est - cross(gyrodotf,rpos) - cross(gyrod,rv); 
figure
plot(accf(:,3),'black');
hold on
plot(acc_LLeg_est(:,3),'green');
hold on
plot(acc_RLeg_est(:,3),'red');

%% Save Data
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
