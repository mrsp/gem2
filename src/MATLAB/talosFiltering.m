%% Samping Rate
fs = 100;
start = 1000;
endl = 5000;
%% Filter the Angular Velocity of Leg IMUs
for i = 1 : dlen
   lgyroB(i,:) = (Rotwb{i}' * lgyroW(i,:)')';
   rgyroB(i,:) = (Rotwb{i}' * rgyroW(i,:)')';
end
fc = 5;
lgyroBf = butterworth2(lgyroB,fc,fs);
rgyroBf  = butterworth2(rgyroB,fc,fs);
%% Filter the Angular Velocity Measurements.
fc = 5;
lwf = butterworth2(lw,fc,fs);
rwf  = butterworth2(rw,fc,fs);
fc = 5;
gyrof = butterworth2(gyro,fc,fs);
gyrodotf = diff([0,0,0;gyrof]*fs);

lwf_est = lgyroBf - gyrof;
rwf_est = rgyroBf - gyrof;
figure
plot(lwf(start:endl,3),'black');
hold on
plot(lwf_est(start:endl,3),'blue');


figure
plot(gyrof(start:endl,2),'black','linewidth',1);
hold on
plot(-rwf_est(start:endl,2),'red');
hold on
plot(-lwf_est(start:endl,2),'green');
%% Filter the Linear Acceleration Measurements
fc = 4.5;
accf = butterworth2(acc,fc,fs);
%%Linear Velocities
%Left-Leg
fc = 4.5;
lvf = butterworth2(lv,fc,fs);
laf = [0 0 0;diff(lvf)*fs];
%Right-Leg
rvf = butterworth2(rv,fc,fs);
raf = [0 0 0;diff(rvf)*fs];
lposf = butterworth2(lpos,fc,fs);
rposf = butterworth2(rpos,fc,fs);
acc_LLeg = -laf - cross(gyrodotf,lpos) - cross(gyrof,lv); 
acc_RLeg = -raf - cross(gyrodotf,rpos) - cross(gyrof,rv); 
fc = 3;
acc_LLeg= butterworth2(acc_LLeg,fc,fs);
acc_RLeg= butterworth2(acc_RLeg,fc,fs);
% 
for i = 1 : dlen
   laccB(i,:) = (Rotwb{i}' * laccW(i,:)')';
   raccB(i,:) = (Rotwb{i}' * raccW(i,:)')';

end
% fc = 4.5;
% laccBf= butterworth2(laccB,fc,fs);
% raccBf= butterworth2(raccB,fc,fs);
% laB = laccBf - accf - cross(gyrodotf,lpos) - cross(gyrof,lv);  
% raB = raccBf - accf - cross(gyrodotf,rpos) - cross(gyrof,rv);  
% acc_LLeg = -laB - cross(gyrodotf,lpos) - cross(gyrof,lv); 
% acc_RLeg = -raB - cross(gyrodotf,rpos) - cross(gyrof,rv); 


%acc_RLeg = accWf + (cross(gyrodotWf,rposW) + cross(gyroWf,cross(gyroWf,rposW)) + 2.0*cross(gyroWf,rvWf) + raWf);
%acc_LLeg = accWf + (cross(gyrodotWf,lposW) + cross(gyroWf,cross(gyroWf,lposW)) + 2.0*cross(gyroWf,lvWf) + laWf);
figure
plot(accf(start:endl,1),'black');
hold on
plot(acc_LLeg(start:endl,1),'green');
hold on
plot(acc_RLeg(start:endl,3),'red');

%% Save Data
if(saveData == 1)
    %LLeg Label
    dlmwrite(strcat(saveDir,'/baccX_LL.txt'),acc_LLeg(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_LL.txt'),acc_LLeg(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_LL.txt'),acc_LLeg(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/baccX_RL.txt'),acc_RLeg(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY_RL.txt'),acc_RLeg(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ_RL.txt'),acc_RLeg(1:dlen,3))
    dlmwrite(strcat(saveDir,'/baccX.txt'),accf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/baccY.txt'),accf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/baccZ.txt'),accf(1:dlen,3))
    
    dlmwrite(strcat(saveDir,'/bgX_LL.txt'),-lwf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY_LL.txt'),-lwf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ_LL.txt'),-lwf(1:dlen,3))
    %RLeg Label
    dlmwrite(strcat(saveDir,'/bgX_RL.txt'),-rwf(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY_RL.txt'),-rwf(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ_RL.txt'),-rwf(1:dlen,3))
    dlmwrite(strcat(saveDir,'/bgX.txt'),gyrof(1:dlen,1))
    dlmwrite(strcat(saveDir,'/bgY.txt'),gyrof(1:dlen,2))
    dlmwrite(strcat(saveDir,'/bgZ.txt'),gyrof(1:dlen,3))
end
