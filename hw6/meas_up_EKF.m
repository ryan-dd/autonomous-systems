function [mu,Sig] = meas_up_EKF(X,m,mu,Sig,N,j,flag,sig)
% This function performs the measurement update corresponding to a
% specific landmark m. See lines 9-20 of Table 7.2 in Probabilistic
% Robotics by Thrun, et al.
 x = X(1); % true states used to create measurements
 y = X(2);
 th = X(3);
 mx = m(1); % true landmark location used to create measurements
 my = m(2);
 mu_x = mu(1); % estimated states to be updated by measurement
 mu_y = mu(2);
 mu_th = mu(3);
 mu_mx = mu(3+2*j-1); % estimated landmark locations
 mu_my = mu(3+2*j);
 sig_r = sig(1); % s.d. of noise levels on measurements
 sig_ph = sig(2);
 % Calculate measurement based on true states (pose, landmark) and noise
 % Measurements: truth + noise
 range = sqrt((mx-x).^2 + (my-y).^2) + sig_r*randn;
 bearing = atan2(my-y,mx-x) - th + sig_ph*randn;
 z = [range; bearing];
 % if landmark detected for first time, initialize landmark location
 % estimate with measurment data
 if flag == 1
 mu_mx = mu_x + range*cos(bearing + mu_th);
 mu_my = mu_y + range*sin(bearing + mu_th);

 mu(3+2*j-1) = mu_mx;
 mu(3+2*j) = mu_my;
 end

 % Calculate predicted measurement based on state estimate
 dx = mu_mx-mu_x;
 dy = mu_my-mu_y;
 q = dx^2 + dy^2;
 zhat = zeros(2,1);
 zhat(1) = sqrt(q);
 zhat(2) = atan2(dy,dx) - mu_th;
 % Jacobian of measurement function wrt state
 Fxj = zeros(5,3+2*N);
 Fxj(1,1) = 1;
 Fxj(2,2) = 1;
 Fxj(3,3) = 1;
 Fxj(4,3+2*j-1) = 1;
 Fxj(5,3+2*j) = 1;
 HH = (1/q)*[-sqrt(q)*dx -sqrt(q)*dy 0 sqrt(q)*dx sqrt(q)*dy; ...
 dy -dx -q -dy dx ];
 H = HH*Fxj;
 % Total uncertainty in predicted measurement
 Q = diag([sig_r^2, sig_ph^2]);
 S = H*Sig*H' + Q;
 % Kalman gain
 K = (Sig*H')/S;
 % Measurement update
 res = z - zhat;
 if (res(2) > pi)
 res(2) = res(2) - 2*pi;
 elseif (res(2) <= -pi)
 res(2) = res(2) + 2*pi;
 end
 Md = res'/S*res;
 if Md < 3^2
 update = K*res;
 mu = mu + update;
 Sig = (eye(3+2*N) - K*H)*Sig*(eye(3+2*N) - K*H)' + K*Q*K'; % Joseph's form
 end
end