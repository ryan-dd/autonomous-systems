% Ground robot simultaneous localization and mapping using FastSLAM from
% Probabilistic Robotics, Thrun et al., Table 13.1
clear all;
dt = 0.1;
tfinal = 40;
t = 0:dt:tfinal;
N = length(t);
% Initial conditions
x_tr0 = 0;
y_tr0 = 5;
th_tr0 = 0;
% Landmark (feature) locations
mx = [6 -7 6 0 -8 1 4 -5]; % x-coordinate of landmarks
my = [4 8 -1 -9 -5 1 -4 3]; % y-coordinate of landmarks
m = [mx; my];
MM = length(mx); % number of landmarks
% Measurement noise parameters
sig_r = 0.1;
sig_ph = 0.05;
Q_t = diag([sig_r^2; sig_ph^2]);
% Motion input plus noise model
v_c = 1 + 0.1*sin(2*pi*0.3*t); % defining control command (noise free)
om_c = -0.2 + 0.1*cos(2*pi*0.2*t);
u_c = [v_c; om_c];
alph1 = 0.1;
alph2 = 0.01;
alph3 = 0.01;
alph4 = 0.1;
alph5 = 0.001;
alph6 = 0.001;
alpha = [alph1 alph2 alph3 alph4 alph5 alph6];
% definining true control command (commanded plus motion noise)
v_tr = v_c + sqrt(alph1*v_c.^2+alph2*om_c.^2).*randn(1,N);
om_tr = om_c + sqrt(alph3*v_c.^2+alph4*om_c.^2).*randn(1,N);
gam_tr = sqrt(alph5*v_c.^2+alph6*om_c.^2).*randn(1,N);
u_tr = [v_tr; om_tr];
x_tr = zeros(1,N);
y_tr = zeros(1,N);
th_tr = zeros(1,N);
x_tr(1) = x_tr0;
y_tr(1) = y_tr0;
th_tr(1) = th_tr0;
% Draw robot at time step 1
% Calculate measurement truth data at time step 1
for j=1:MM
 z_tr = meas_truth([x_tr(1);y_tr(1);th_tr(1)],m(:,j),Q_t);
 range_tr(1,j) = z_tr(1);
 bearing_tr(1,j) = z_tr(2);
end
% Create pose truth and measurement truth data
for i = 2:N
 x_tr(i) = x_tr(i-1) + (-v_tr(i)/om_tr(i)*sin(th_tr(i-1)) + v_tr(i)/om_tr(i)*sin(th_tr(i-1)+om_tr(i)*dt));
 y_tr(i) = y_tr(i-1) + (v_tr(i)/om_tr(i)*cos(th_tr(i-1)) - v_tr(i)/om_tr(i)*cos(th_tr(i-1)+om_tr(i)*dt));
 th_tr(i) = th_tr(i-1) + om_tr(i)*dt +gam_tr(i)*dt;
 X_i = [x_tr(i); y_tr(i); th_tr(i)];
 % drawRobot(x_tr(i),y_tr(i),th_tr(i),m,t(i));
 % pause(0.05);
 % Calculate measurement truth data at time step i
 for j=1:MM
 z_tr = meas_truth(X_i,m(:,j),Q_t);
 range_tr(i,j) = z_tr(1);
 bearing_tr(i,j) = z_tr(2);
 end
end
X_tr = [x_tr; y_tr; th_tr]; % matrix of true state vectors at all times
% Particles are made up of robot pose states (x, y, heading), x and y
% locations of MM landmarks, 2x2 covariance matrices for MM landmarks, and
% weight associated with the particle
% Initialize particle set to be at the origin
K = 1000; % number of particles
% for k=1:K
% Y(k).X = [x_tr0; y_tr0; th_tr0];
% Y(k).Mu_lm = zeros(2,MM);
% Y(k).Sig_lm = zeros(2,2,MM);
% Y(k).w = 1/K;
% end
X_sv = zeros(3,K,N);
Mu_lm_sv = zeros(2,MM,K,N);
Sig_lm_sv = zeros(2,2,MM,K,N);
w_sv = ones(K,N)/K;
% initialize all particles to be in the initial pose at the first time step
for k=1:K
 X_sv(:,k,1) = [x_tr0; y_tr0; th_tr0];
 for mm=1:MM
 Mu_lm_sv(:,mm,k,1) = [0; 0];
 Sig_lm_sv(:,:,mm,k,1) = 100*eye(2);
 end
end
% Initialize estimates for time 1
X_t = squeeze(X_sv(:,:,1));
Mu_lm_t = squeeze(Mu_lm_sv(:,:,:,1));
Sig_lm_t = squeeze(Sig_lm_sv(:,:,:,:,1));
w_t = squeeze(w_sv(:,1));
% Draw robot particles
px = X_t(1,:);
py = X_t(2,:);
%%%%drawRobotParticles(x_tr0,y_tr0,th_tr0,m,px,py,t(1));
% Fast SLAM algorithm : loop through the data
for i=2:N
 z_t = [range_tr(i,:); bearing_tr(i,:)];
 u_t = u_c(:,i);
 X_t_1 = X_t;
 Mu_lm_t_1 = Mu_lm_t;
 Sig_lm_t_1 = Sig_lm_t;
 w_t_1 = w_t;

 % Calculate landmark index (which landmark are we observing on this
 % time step)
 l = 1 + mod(i,MM);

 % Call Fast SLAM algorithm
 [X_t,Mu_lm_t,Sig_lm_t,k_max] = Fast_SLAM_alg_mult_lm(z_t,u_t,X_t_1,Mu_lm_t_1,Sig_lm_t_1,l,Q_t,alpha,dt,i);
 X_sv(:,:,i) = X_t;
 Mu_lm_sv(:,:,:,i) = Mu_lm_t;
 Sig_lm_sv(:,:,:,:,i) = Sig_lm_t;
 k_max_sv(i) = k_max;

% % Use highest-weighted particle for belief
% mu_x(i) = X_t(1,k_max);
% mu_y(i) = X_t(2,k_max);
% mu_th(i) = X_t(3,k_max);

 % Use mean of particle cloud for belief
 mu_x(i) = mean(X_t(1,:));
 mu_y(i) = mean(X_t(2,:));
 mu_th(i) = mean(X_t(3,:));

 % Draw particles and landmarks during estimation
 lm1_x = mean(squeeze(Mu_lm_sv(1,1,:,i)));
 lm1_y = mean(squeeze(Mu_lm_sv(2,1,:,i)));
 lm2_x = mean(squeeze(Mu_lm_sv(1,2,:,i)));
 lm2_y = mean(squeeze(Mu_lm_sv(2,2,:,i)));
 lm3_x = mean(squeeze(Mu_lm_sv(1,3,:,i)));
 lm3_y = mean(squeeze(Mu_lm_sv(2,3,:,i)));
 lm4_x = mean(squeeze(Mu_lm_sv(1,4,:,i)));
 lm4_y = mean(squeeze(Mu_lm_sv(2,4,:,i)));
 lm5_x = mean(squeeze(Mu_lm_sv(1,5,:,i)));
 lm5_y = mean(squeeze(Mu_lm_sv(2,5,:,i)));
 lm6_x = mean(squeeze(Mu_lm_sv(1,6,:,i)));
 lm6_y = mean(squeeze(Mu_lm_sv(2,6,:,i)));
 lm7_x = mean(squeeze(Mu_lm_sv(1,7,:,i)));
 lm7_y = mean(squeeze(Mu_lm_sv(2,7,:,i)));
 lm8_x = mean(squeeze(Mu_lm_sv(1,8,:,i)));
 lm8_y = mean(squeeze(Mu_lm_sv(2,8,:,i)));
 lm_data = [lm1_x lm2_x lm3_x lm4_x lm5_x lm6_x lm7_x lm8_x; ...
 lm1_y lm2_y lm3_y lm4_y lm5_y lm6_y lm7_y lm8_y];
 px = X_t(1,:);
 py = X_t(2,:);
 %%%%%%drawRobotParticlesLM(x_tr(i),y_tr(i),th_tr(i),m,lm_data,px,py,t(i));
 pause(0.01);
end
figure(2); clf;
subplot(311);
plot(t,x_tr,t,mu_x);
ylabel('x position (m)');
legend('true','estimated','Location','NorthWest');
subplot(312);
plot(t,y_tr,t,mu_y);
ylabel('y position (m)')
subplot(313);
plot(t,180/pi*th_tr,t,180/pi*mu_th);
xlabel('time (s)');
ylabel('heading (deg)');
% % landmark locations based on highest weighted particle
% lm1_x = squeeze(Mu_lm_sv(1,1,k_max,:));
% lm1_y = squeeze(Mu_lm_sv(2,1,k_max,:));
% lm2_x = squeeze(Mu_lm_sv(1,2,k_max,:));
% lm2_y = squeeze(Mu_lm_sv(2,2,k_max,:));
% lm3_x = squeeze(Mu_lm_sv(1,3,k_max,:));
% lm3_y = squeeze(Mu_lm_sv(2,3,k_max,:));
% lm4_x = squeeze(Mu_lm_sv(1,4,k_max,:));
% lm4_y = squeeze(Mu_lm_sv(2,4,k_max,:));
% lm5_x = squeeze(Mu_lm_sv(1,5,k_max,:));
% lm5_y = squeeze(Mu_lm_sv(2,5,k_max,:));
% lm6_x = squeeze(Mu_lm_sv(1,6,k_max,:));
% lm6_y = squeeze(Mu_lm_sv(2,6,k_max,:));
% lm7_x = squeeze(Mu_lm_sv(1,7,k_max,:));
% lm7_y = squeeze(Mu_lm_sv(2,7,k_max,:));
% lm8_x = squeeze(Mu_lm_sv(1,8,k_max,:));
% lm8_y = squeeze(Mu_lm_sv(2,8,k_max,:));
% landmark locations based on mean of particle locations
lm1_x = mean(squeeze(Mu_lm_sv(1,1,:,:)));
lm1_y = mean(squeeze(Mu_lm_sv(2,1,:,:)));
lm2_x = mean(squeeze(Mu_lm_sv(1,2,:,:)));
lm2_y = mean(squeeze(Mu_lm_sv(2,2,:,:)));
lm3_x = mean(squeeze(Mu_lm_sv(1,3,:,:)));
lm3_y = mean(squeeze(Mu_lm_sv(2,3,:,:)));
lm4_x = mean(squeeze(Mu_lm_sv(1,4,:,:)));
lm4_y = mean(squeeze(Mu_lm_sv(2,4,:,:)));
lm5_x = mean(squeeze(Mu_lm_sv(1,5,:,:)));
lm5_y = mean(squeeze(Mu_lm_sv(2,5,:,:)));
lm6_x = mean(squeeze(Mu_lm_sv(1,6,:,:)));
lm6_y = mean(squeeze(Mu_lm_sv(2,6,:,:)));
lm7_x = mean(squeeze(Mu_lm_sv(1,7,:,:)));
lm7_y = mean(squeeze(Mu_lm_sv(2,7,:,:)));
lm8_x = mean(squeeze(Mu_lm_sv(1,8,:,:)));
lm8_y = mean(squeeze(Mu_lm_sv(2,8,:,:)));
figure(4); clf;
subplot(421); plot(t,lm1_x,'r',t,lm1_y,'b',[0 tfinal],[mx(1) mx(1)],'r--',[0 tfinal],[my(1) my(1)],'b--');
title('landmark 1'); xlabel('time (s)'); ylabel('position (m)');
subplot(422); plot(t,lm2_x,'r',t,lm2_y,'b',[0 tfinal],[mx(2) mx(2)],'r--',[0 tfinal],[my(2) my(2)],'b--');
title('landmark 2'); xlabel('time (s)'); ylabel('position (m)');
subplot(423); plot(t,lm3_x,'r',t,lm3_y,'b',[0 tfinal],[mx(3) mx(3)],'r--',[0 tfinal],[my(3) my(3)],'b--');
title('landmark 3'); xlabel('time (s)'); ylabel('position (m)');
subplot(424); plot(t,lm4_x,'r',t,lm4_y,'b',[0 tfinal],[mx(4) mx(4)],'r--',[0 tfinal],[my(4) my(4)],'b--');
title('landmark 4'); xlabel('time (s)'); ylabel('position (m)');
subplot(425); plot(t,lm5_x,'r',t,lm5_y,'b',[0 tfinal],[mx(5) mx(5)],'r--',[0 tfinal],[my(5) my(5)],'b--');
title('landmark 5'); xlabel('time (s)'); ylabel('position (m)');
subplot(426); plot(t,lm6_x,'r',t,lm6_y,'b',[0 tfinal],[mx(6) mx(6)],'r--',[0 tfinal],[my(6) my(6)],'b--');
title('landmark 6'); xlabel('time (s)'); ylabel('position (m)');
subplot(427); plot(t,lm7_x,'r',t,lm7_y,'b',[0 tfinal],[mx(7) mx(7)],'r--',[0 tfinal],[my(7) my(7)],'b--');
title('landmark 7'); xlabel('time (s)'); ylabel('position (m)');
subplot(428); plot(t,lm8_x,'r',t,lm8_y,'b',[0 tfinal],[mx(8) mx(8)],'r--',[0 tfinal],[my(8) my(8)],'b--');
title('landmark 8'); xlabel('time (s)'); ylabel('position (m)');
function [X_t,Mu_lm_t,Sig_lm_t,k_max] = Fast_SLAM_alg_mult_lm(z_t,u_t,X_t_1,Mu_lm_t_1,Sig_lm_t_1,l,Q_t,alpha,dt,t_index)
 % Fast_SLAM_alg.m
 % Fast SLAM algorithm
 % Uses low variance sampler from Table 13.1
% persistent Mu_lm_t Mu_lm_t_1 Sig_lm_t Sig_lm_t_1

 % Initialize vectors, matrix
 [~,M,K] = size(Mu_lm_t_1);
% X_t = zeros(size(X_t_1));
 Mu_lm_t = Mu_lm_t_1;
 Sig_lm_t = Sig_lm_t_1;
% w_t = zeros(size(w_t_1));

 % Motion prediction for each particle
 for k = 1:K
 X_t(:,k) = samp_motion_model(u_t,X_t_1(:,k),alpha,dt);

 for l=1:M
 Mu_t_1 = squeeze(Mu_lm_t_1(:,l,k));
 Sig_t_1 = squeeze(Sig_lm_t_1(:,:,l,k));
 % Calculate particle weight and landmark location estimate for selected landmark
 if t_index == 2
 % Call function to initialize landmark location at first
 [Mu_lm_t(:,l,k),Sig_lm_t(:,:,l,k),w_t(k)] = initialize_lm(X_t(:,k),z_t(:,l),Q_t);
 else
 % Call function to calculate landmark locations and importance
 % factor for each particle
 [Mu_lm_t(:,l,k),Sig_lm_t(:,:,l,k),w_t(k)] = meas_model(X_t(:,k),z_t(:,l),Mu_t_1,Sig_t_1,Q_t);
 end
 end
 end

% % Plot particles after sampling motion model
% mu_x = mean(Mu_lm_t(1,1,:));
% mu_y = mean(Mu_lm_t(2,1),:);
%
% x_tr = X_t(1,:);
% y_tr = X_t(2,:);
% th_tr = X_t(3,:);
%
% px = Mu_lm_t(1,:);
% py = Mu_lm_t(2,:);
%
% drawRobotParticles(mu_x,mu_y,mu_th,m,px,py,1);
% drawRobotParticles(x_tr,y_tr,th_tr,m,px,py,1);
% pause(0.1);
 % LV resampling algorithm assumes weights sum to 1.
 % Must normalize weights
 w_t = w_t/sum(w_t);
 [w_max,k_max] = max(w_t);
 % Resample particles using low-variance resampler
 [X_t,w_t,ind] = LVsamp(X_t,w_t,K);

 % Change values for Mu_lm_t and Sig_lm_t based on resampling
 Mu_ = Mu_lm_t;
 Sig_ = Sig_lm_t;
 for k=1:K
 Mu_lm_t(:,:,k) = Mu_(:,:,ind(k));
 Sig_lm_t(:,:,:,k) = Sig_(:,:,:,ind(k));
 end


% % Plot particles after resampling based on measurement
% mu_x = mean(Chi_t(1,:));
% mu_y = mean(Chi_t(2,:));
% mu_th = mean(Chi_t(3,:));
%
% px = Chi_t(1,:);
% py = Chi_t(2,:);
%
% drawRobotParticles(x_tr,y_tr,th_tr,m,px,py,1);
% pause(0.1);
end
function x_t = samp_motion_model(u_t,x_t_1,alpha,dt)
 % Sample the motion model utilizing the algorithm in Table 5.3

 % inputs at current time
 v = u_t(1);
 om = u_t(2);
 % standard deviation of input noise
 sd_v = sqrt(alpha(1)*v^2 + alpha(2)*om^2);
 sd_om = sqrt(alpha(3)*v^2 + alpha(4)*om^2);
 sd_gam = sqrt(alpha(5)*v^2 + alpha(6)*om^2);
 % sampled inputs
 vhat = v + sd_v*randn(1);
 omhat = om + sd_om*randn(1);
 gamhat = sd_gam*randn(1);
 % particle state at prior time
 x = x_t_1(1);
 y = x_t_1(2);
 th = x_t_1(3);

 % particle state at current time
 xpr = x + (-vhat/omhat*sin(th) + vhat/omhat*sin(th+omhat*dt));
 ypr = y + (vhat/omhat*cos(th) - vhat/omhat*cos(th+omhat*dt));
 thpr = th + omhat*dt + gamhat*dt;

 x_t = [xpr; ypr; thpr];

end
function [mu_t,Sig_t,w_t] = initialize_lm(X_t,z_t,Q_t)
 % Initialize landmark location first time that it is seen

 % robot pose estimate
 x = X_t(1);
 y = X_t(2);
 th = X_t(3);

 % range and bearing measurements
 r = z_t(1);
 ph = z_t(2);

 % landmark initial x and y location
 mu_x = x + r*cos(ph + th);
 mu_y = y + r*sin(ph + th);
 mu_t = [mu_x; mu_y];

 q = (mu_x-x)^2 + (mu_y-y)^2;

 % Jacobian of measurement function wrt landmark location (x,y)
 %**** check this Jacobian ****
 H = zeros(2,2);
 H(1,1) = (mu_x-x)/sqrt(q);
 H(1,2) = (mu_y-y)/sqrt(q);
 H(2,1) = -(mu_y-y)/q;
 H(2,2) = (mu_x-x)/q;

 HT = H';
 Sig_t = H\Q_t/HT;

 w_t = 1/1000;

end
function [mu_t,Sig_t,w_t] = meas_model(X_t,z_t,mu_t_1,Sig_t_1,Q_t)
 % Calculate estimate of landmark location for particle and particle weight

 % landmark location estimate
 mx = mu_t_1(1);
 my = mu_t_1(2);

 % particle state
 xp = X_t(1);
 yp = X_t(2);
 thp = X_t(3);

 % range and bearing measurements
 r = z_t(1);
 ph = z_t(2);

 % estimates of range and bearing from particle to landmark
 q = (mx-xp)^2 + (my-yp)^2;
 r_hat = sqrt(q);
 ph_hat = wrapToPi(atan2(my-yp,mx-xp) - thp);

 residual = [r-r_hat; wrapToPi(ph-ph_hat)];

 % Jacobian of measurement function wrt landmark location (x,y)
 H = zeros(2,2);
 H(1,1) = (mx-xp)/sqrt(q);
 H(1,2) = (my-yp)/sqrt(q);
 H(2,1) = -(my-yp)/q;
 H(2,2) = (mx-xp)/q;

 Q = H*Sig_t_1*H' + Q_t;
 K = Sig_t_1*H'/Q;

 mu_t = mu_t_1 + K*residual;
 Sig_t = (eye(2) - K*H)*Sig_t_1;

 w_t = (det(2*pi*Q))^(-0.5)*exp(-0.5*residual'/Q*residual);

end
function [X,w,ind] = LVsamp(X_bar,w_bar,M)
 % LVsamp.m
 %
 % Low-variance sampler according to Table 4.4 in Probabilistic Robotics text
 n = 3; % number of states

 x_bar = X_bar;

 x = [];
 w = [];
 ind = [];

 r = rand/M;
 c = w_bar(1);
 i = 1;
 for m = 1:M
 U = r+(m-1)/M;
 while U > c
 i = i + 1;
 c = c + w_bar(i);
 end
 x = [x x_bar(:,i)];
 w = [w w_bar(i)];
 ind = [ind i];
 end

 % Combating particle deprivation
 % Important! Not part of standard algorithm
 P = cov(x_bar'); % covariance of prior

 uniq = length(unique(ind)); % number of unique particles in resampled cloud
 if uniq/M < 0.5 % if there is a lot of duplication
 Q = P/((M*uniq)^(1/n)); % add noise to the samples
 x = x + Q*randn(size(x));
 end

% wtmp = w;

 % reset weights to make them uniform
 w = ones(1,M)/M;

 % particles after resampling
 X = x;

% figure(5);
% plot(x_bar(1,:),w_bar,'o',x(1,:),w,'o');
% axis([-10 10 0 1]);
% legend('weights','resampled');
% plot(wtmp);
% pause(0.1);
end
function z_tr = meas_truth(truth,landmark,Q_t)
true_x = truth(1);
true_y = truth(2);
true_theta = truth(3);
f_x = landmark(1);
f_y = landmark(2);
q = (f_x - true_x)^2 + (f_y - true_y)^2;

phi = atan2((f_y - true_y), (f_x - true_x)) - true_theta;
phi = wrapToPi(phi);
r = sqrt(q);
z_tr = [r + normrnd( 0, Q_t(1,1)); wrapToPi(phi+normrnd( 0, Q_t(2,2)))];
end