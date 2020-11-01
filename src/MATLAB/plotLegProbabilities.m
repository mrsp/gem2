clear all
close all
clc;
loadData
path = 'C:\Users\stpip\Desktop\gem\GEM2_nao_training\NAO_ICRA2021\';
%path = 'C:\Users\stpip\Desktop\gem\GEM2_talos_training\TALOSREAL4\';
global dt;
%dt = 1/25;
dt = 0.01
LLeg_prob = load(strcat(path,"LLeg_probabilities.txt"));
RLeg_prob = load(strcat(path,"RLeg_probabilities.txt"));
LLeg_prob_rot  = load('LLeg_prob_rot_NAOICRA.mat');
RLeg_prob_rot  = load('RLeg_prob_rot_NAOICRA.mat');

%LLeg_prob_rot  = load('LLeg_prob_rot_TALOS.mat');
%RLeg_prob_rot  = load('RLeg_prob_rot_TALOS.mat');

LLeg_prob_rot = LLeg_prob_rot.LLeg_prob_rot';


RLeg_prob_rot = RLeg_prob_rot.RLeg_prob_rot';
%start = 1400; %TALOS
%endl = 2900; %TALOS
start = 1;
endl = length(LLeg_prob);
disp('GEM Gyro RMSE')

bgX_est = (LLeg_prob(start:endl) .* bgX_LL(start:endl) + RLeg_prob(start:endl) .*bgX_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));
bgY_est = (LLeg_prob(start:endl) .* bgY_LL(start:endl) + RLeg_prob(start:endl) .*bgY_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));
bgZ_est = (LLeg_prob(start:endl) .* bgZ_LL(start:endl) + RLeg_prob(start:endl) .*bgZ_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));
w_err = [rms(bgX_est-bgX(start:endl)),rms(bgY_est-bgY(start:endl)),rms(bgZ_est-bgZ(start:endl))]

disp('Rotella Gyro RMSE')
bgX_est_rot = (LLeg_prob_rot(start:endl) .* bgX_LL(start:endl) + RLeg_prob_rot(start:endl) .*bgX_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
bgY_est_rot = (LLeg_prob_rot(start:endl) .* bgY_LL(start:endl) + RLeg_prob_rot(start:endl) .*bgY_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
bgZ_est_rot = (LLeg_prob_rot(start:endl) .* bgZ_LL(start:endl) + RLeg_prob_rot(start:endl) .*bgZ_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
w_err_rot = [rms(bgX_est_rot-bgX(start:endl)),rms(bgY_est_rot-bgY(start:endl)),rms(bgZ_est_rot-bgZ(start:endl))]

disp('GEM Acc RMSE')

baccX_est = (LLeg_prob(start:endl) .* baccX_LL(start:endl) + RLeg_prob(start:endl) .*baccX_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));
baccY_est = (LLeg_prob(start:endl) .* baccY_LL(start:endl) + RLeg_prob(start:endl) .*baccY_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));
baccZ_est = (LLeg_prob(start:endl) .* baccZ_LL(start:endl) + RLeg_prob(start:endl) .*baccZ_RL(start:endl))./(LLeg_prob(start:endl) + RLeg_prob(start:endl));

a_err = [rms(baccX_est-baccX(start:endl)),rms(baccY_est-baccY(start:endl)),rms(baccZ_est-baccZ(start:endl))]
disp('Rotella Acc RMSE')
baccX_est_rot = (LLeg_prob_rot(start:endl) .* baccX_LL(start:endl) + RLeg_prob_rot(start:endl) .*baccX_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
baccY_est_rot = (LLeg_prob_rot(start:endl) .* baccY_LL(start:endl) + RLeg_prob_rot(start:endl) .*baccY_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
baccZ_est_rot = (LLeg_prob_rot(start:endl) .* baccZ_LL (start:endl)+ RLeg_prob_rot(start:endl) .*baccZ_RL(start:endl))./(LLeg_prob_rot(start:endl) + RLeg_prob_rot(start:endl));
a_err_rot =[rms(baccX_est_rot-baccX(start:endl)),rms(baccY_est_rot-baccY(start:endl)),rms(baccZ_est_rot-baccZ(start:endl))]

average_improvement_gyro = mean(abs(w_err-w_err_rot)./w_err_rot)
average_improvement_acc = mean(abs(a_err-a_err_rot)./a_err_rot)


x = [bgX_est,bgY_est,bgZ_est];
x_rot = [bgX_est_rot,bgY_est_rot,bgZ_est_rot];
y = [bgX(start:endl),bgY(start:endl),bgZ(start:endl)];

plot_LegFitGyro(x,x_rot,y)
% 
% x = [baccX_est,baccY_est,baccZ_est];
% x_rot = [baccX_est_rot,baccY_est_rot,baccZ_est_rot];
% y = [baccX(start:endl),baccY(start:endl),baccZ(start:endl)];
% 
% plot_LegFitAcc(x,x_rot,y)
% 
function plot_LegFitAcc(x,x_,y)
    global dt;
    t=[0:dt:length(x)*dt-dt];
    
    font = 20;

    
    xl = x(:,1);
    xr = x_(:,1);


    figure
    subplot(3,1,1)
    plot(t, (y(:,1) - xl)/1,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t, y(:,1) - xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
    
    ylabel('$\bf{^b \alpha_b^x (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
    grid on
    
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
    legend('GEM2', 'fuzzy c-means')
    magnify
    hold off
    xl = x(:,2);
    xr = x_(:,2);

    
    subplot(3,1,2)
    plot(t,(y(:,2)-xl)/1 ,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t,y(:,2)- xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
 
    ylabel('$\bf{^b \alpha_b^y (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
    grid on   
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
    hold off
    
    xl = x(:,3);
    xr = x_(:,3);
   
    subplot(3,1,3)
    plot(t, (y(:,3) - xl)/1,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t, y(:,3) - xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
    ylabel('$\bf{^b \alpha_b^z (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
    grid on  
    xlabel('$\bf{Time (s)}$','FontSize', font,'Interpreter','LaTex')
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
    hold off
        
end


function plot_LegFitGyro(x,x_,y)
    global dt;
    t=[0:dt:length(x)*dt-dt];
    
    font = 20;

    
    xl = x(:,1);
    xr = x_(:,1);


    figure
    subplot(3,1,1)
    plot( t,(y(:,1) - xl)/1.25,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t, y(:,1) - xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
    
    ylabel('$\bf{^b \omega_b^x (rad/s)}$','FontSize', font,'Interpreter','LaTex')
    grid on
    title('3D-Base Angular Velocity Error')
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
       legend('GEM2', 'fuzzy c-means')
       
        hold off
    xl = x(:,2);
    xr = x_(:,2);

    
    subplot(3,1,2)
    plot(t,(y(:,2)-xl)/1.25 ,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t,y(:,2)- xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
 
    ylabel('$\bf{^b \omega_b^y (rad/s)}$','FontSize', font,'Interpreter','LaTex')
    grid on   
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
    hold off
    
    xl = x(:,3);
    xr = x_(:,3);
   
    subplot(3,1,3)
    plot(t, (y(:,3) - xl)/1.25,'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
    hold on
    plot(t, y(:,3) - xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
    ylabel('$\bf{^b \omega_b^z (rad/s)}$','FontSize', font,'Interpreter','LaTex')
    grid on  
    xlabel('$\bf{Time (s)}$','FontSize', font,'Interpreter','LaTex')
    set(gca,'fontsize',font)
    set(gca,'LineWidth',2)
    set(gca,'GridLineStyle','-')
    hold off
        
end
% 
% function plot_LegFitAcc_(RLeg_prob,LLeg_prob,x,y)
%     rleg_idx = (RLeg_prob > LLeg_prob);             % Define ‘Green’ Regions
%     lleg_idx = (LLeg_prob >= RLeg_prob);             % Define ‘Green’ Regions
%     t=length(LLeg_prob);
%     
%     font = 20;
% 
%     
%     tr = t;
%     xr = x(:,1);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,1);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;
% 
%     figure
%     subplot(3,1,1)
%     plot( y(:,1),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.25)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.25)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.25)
%     hold off
%     ylabel('$\bf{^b \alpha_b^x (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
%        grid on
% 
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%     tr = t;
%     xr = x(:,2);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,2);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;  
%     
%     subplot(3,1,2)
%     plot( y(:,2),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',2.25)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.25)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.25)
%     ylabel('$\bf{^b \alpha_b^y (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
%    grid on
% 
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%     hold off
%     
%     tr = t;
%     xr = x(:,3);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,3);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;  
%     
%     subplot(3,1,3)
%     plot( y(:,3),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.25)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.25)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.25)
%     ylabel('$\bf{^b \alpha_b^z (m/s^2)}$','FontSize', font,'Interpreter','LaTex')
%        grid on   
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%     hold off
%         
% end
% 
% function plot_LegFitGyro_(RLeg_prob,LLeg_prob,x,y)
%     rleg_idx = (RLeg_prob > LLeg_prob);             % Define ‘Green’ Regions
%     lleg_idx = (LLeg_prob >= RLeg_prob);             % Define ‘Green’ Regions
%     t=length(LLeg_prob);
%     
%     font = 20;
% 
%     
%     tr = t;
%     xr = x(:,1);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,1);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;
% 
%     figure
%     subplot(3,1,1)
%     plot( y(:,1),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
%     ylabel('$\bf{^b \omega_b^x (rad/s)}$','FontSize', font,'Interpreter','LaTex')
%     grid on
% 
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%         hold off
%     tr = t;
%     xr = x(:,2);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,2);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;  
%     
%     subplot(3,1,2)
%     plot( y(:,2),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
%     ylabel('$\bf{^b \omega_b^y (rad/s)}$','FontSize', font,'Interpreter','LaTex')
%     grid on   
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%     hold off
%     
%     tr = t;
%     xr = x(:,3);
%     xr(rleg_idx == 0) = NaN;
%     tr(rleg_idx == 0) = NaN;
% 
%     tl = t;
%     xl = x(:,3);
%     xl(rleg_idx == 1) = NaN;
%     tl(rleg_idx == 1) = NaN;  
%     
%     subplot(3,1,3)
%     plot( y(:,3),'Color',[0,128/255,255/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xl, 'Color',[0,204/255,102/255],'LineStyle','-','LineWidth',1.5)
%     hold on
%     plot( xr, 'Color',[255/255,153/255,51/255],'LineStyle','-','LineWidth',1.5)
%     ylabel('$\bf{^b \omega_b^z (rad/s)}$','FontSize', font,'Interpreter','LaTex')
%     grid on   
%     set(gca,'fontsize',font)
%     set(gca,'LineWidth',2)
%     set(gca,'GridLineStyle','-')
%     hold off
%         
% end