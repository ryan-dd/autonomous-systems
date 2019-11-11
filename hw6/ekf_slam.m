% robot_EKF_mult_lm.m
%
% EKF localization implementation from Probabilistic Robotics, Thrun et
% al., Table 7.2
%
% This implementation handles multiple landmarks
%
% State variables are (x,y,th)
% State estimates are (mu_x,mu_y,mu_th)
% Estimation error covariance matrix is Sig
clear all;
% load('randvars.mat'); % load a data file with set random noise
 % for debugging
dt = 0.1;
tfinal = 20;
t = 0:dt:tfinal;
Nsteps = length(t);
% Initial conditions
x0 = 0;
y0 = 0;
th0 = 0;
% Motion input plus noise model
v_c = 2.0 + 0.5*sin(2*pi*0.2*t);
om_c = -0.5 + 0.2*cos(2*pi*0.6*t);
alph1 = 0.1;
alph2 = 0.01;
alph3 = 0.01;
alph4 = 0.1;
v = v_c + sqrt(alph1*v_c.^2+alph2*om_c.^2).*randn(1,Nsteps);
om = om_c + sqrt(alph3*v_c.^2+alph4*om_c.^2).*randn(1,Nsteps);
x(1) = 0;
y(1) = 0;
th(1) = 0;
% Landmark (feature) locations
% mx = [6 -7 6 -3 0 ]; % x-coordinate of landmarks
% my = [4 -8 -4 0 2 ]; % y-coordinate of landmarks
% m = [mx; my];
MM = 20; % number of landmarks
m = 20*(rand(2,MM)-0.5);
% Draw robot at time step 1
% drawRobot(x(1),y(1),th(1),m,t(1));
for i = 2:Nsteps
 x(i) = x(i-1) + (-v(i)/om(i)*sin(th(i-1)) + v(i)/om(i)*sin(th(i-1)+om(i)*dt));
 y(i) = y(i-1) + (v(i)/om(i)*cos(th(i-1)) - v(i)/om(i)*cos(th(i-1)+om(i)*dt));
 th(i) = th(i-1) + om(i)*dt;
 %drawRobot(x(i),y(i),th(i),m,t(i));
% pause(0.05);
end
X = [x; y; th]; % matrix of true state vectors at all times
% Localize robot using EKF from Table 7.2
% EKF parameters
sig_r = 0.1;
sig_ph = 0.05;
sig = [sig_r sig_ph];
% Initial conditions of state estimates at time zero
mu = zeros(3+2*MM,1);
Sig = 10^10*diag([0,0,0,ones(1,2*MM)]);
mu_sv = zeros(3+2*MM,Nsteps);
% Process noise in pose dynamics
R = diag([0.2^2 0.2^2 0.05^2]);
% EKF SLAM implementation -- loop through data
for i=2:Nsteps
 % Prediction step
 Th = mu(3); % Use prior theta to predict current states
 % Jacobian of g(u(t),x(t-1)
 Gx = zeros(3,3);
 Gx(1,3) = -v_c(i)/om_c(i)*cos(Th) + v_c(i)/om_c(i)*cos(Th+om_c(i)*dt);
 Gx(2,3) = -v_c(i)/om_c(i)*sin(Th) + v_c(i)/om_c(i)*sin(Th+om_c(i)*dt);
 % State estimate - prediction step
 g1 = (-v_c(i)/om_c(i)*sin(Th) + v_c(i)/om_c(i)*sin(Th+om_c(i)*dt));
 g2 = (v_c(i)/om_c(i)*cos(Th) - v_c(i)/om_c(i)*cos(Th+om_c(i)*dt));
 g3 = om_c(i)*dt;
 g = [g1; g2; g3];
 Fx = [eye(3) zeros(3,2*MM)];

 mubar = mu + Fx'*g;
 G = eye(3+2*MM) + Fx'*Gx*Fx;

 % State covariance - prediction step
 Sigbar = G*Sig*G' + Fx'*R*Fx;
 % Measurement update step
 if i == 2
 flag = 1;
 end

 for j=1:MM
 % If landmark is detected for first time set flag so that landmark
 % estimate will be initialized in measurment update function
 % In simplest case landmark estimates are initialized at first time
 % step (when first seen)
 if i == 2
 flag = 1;
 end
 [mubar,Sigbar] = meas_up_EKF(X(:,i),m(:,j),mubar,Sigbar,MM,j,flag,sig);
 flag = 0;
 end

 mu = mubar;
 Sig = Sigbar;

 mu_sv(:,i) = mu;

end
mu_x = mu_sv(1,:);
mu_y = mu_sv(2,:);
mu_th = mu_sv(3,:);
mu_mx = mu_sv(4:2:3+2*MM,:);
mu_my = mu_sv(5:2:3+2*MM,:);
mx = m(1,:);
my = m(2,:);
figure(2); clf;
subplot(311);
plot(t,x,t,mu_x);
ylabel('x position (m)');
legend('true','estimated','Location','NorthWest');
subplot(312);
plot(t,y,t,mu_y);
ylabel('y position (m)')
subplot(313);
plot(t,180/pi*th,t,180/pi*mu_th);
xlabel('time (s)');
ylabel('heading (deg)');
% figure(3); clf;
% plot(t,x-mu_x,t,y-mu_y,t,th-mu_th);
% ylabel('error (m, rad)');
% legend('x position','y position','heading','Location','NorthEast');
figure(3); clf;
plot([min(t) max(t)],[mx(1:5)' mx(1:5)'],'r--'); hold on;
plot(t,mu_mx(1:5,:),'b-'); hold off;
xlabel('time (s)');
ylabel('landmark x-coordinate estimate (m)');
figure(4); clf;
plot(mx,my,'r+');
xlabel('x (m)');
ylabel('y (m)');
axis([-10 10 -10 10]);
axis('square'); hold on;
for i=1:MM
 plot(mu_mx(:,i),mu_my(:,i),'b.');
 pause(0.5);
end
figure(10); clf;
b = bar3(abs(Sig));
colorbar
for k = 1:length(b)
 zdata = b(k).ZData;
 b(k).CData = zdata;
 b(k).FaceColor = 'interp';
end
caxis([0 0.04]);