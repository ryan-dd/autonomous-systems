% MDP_example.m
%
% Use discrete value iteration to solve simple Markov Decision Process
% path planning problem
%
% Assume robot moves in desired direction with probability 0.8, 90 degrees
% left of desired direction with probability 0.1, and 90 degrees right of
% desired direction with probability 0.1
clear all;
MDP_hw_map;
% Initialize reward map
r = -2*ones(Np,Np);
r = r + 100000*goal - 5000*obs1 - 5000*obs2 - 5000*obs3 - 100*walls;
% Initialize value function and policy
V = zeros(Np,Np);
V = V + 1000*goal;
pol = zeros(Np,Np);
% Initialize motion probabilities
p_straight = 0.8;
p_left = 0.1;
p_right = 0.1;
gamma = 0.995;
del = 100000;
% del_old = 200000;
sumVold = 0;
% figure(1); clf;
% Use backup step to calculate value function
% First calculate max of value function over all actions
% while ((del > 500) && (del_old > del))
while (del > 30)
 del_old = del;
 for i = 2:N+1
 for j = 2:N+1
 if (goal(i,j) == 1)
 % skip this cell
 else
 Vip1j = V(i+1,j);
 Vijm1 = V(i,j-1);
 Vim1j = V(i-1,j);
 Vijp1 = V(i,j+1);

 % north
 Vn = Vijp1*p_straight + Vip1j*p_right + Vim1j*p_left;
 % east
 Ve = Vip1j*p_straight + Vijm1*p_right + Vijp1*p_left;
 % south
 Vs = Vijm1*p_straight + Vim1j*p_right + Vip1j*p_left;
 % west
 Vw = Vim1j*p_straight + Vijp1*p_right + Vijm1*p_left;
 [Vmax,pol(i,j)] = max([Vn Ve Vs Vw]);
 V(i,j) = gamma*(Vmax + r(i,j));
 end
 end
% Vp = flip(V(2:M+1,2:N+1));
% x = [0.5 3.5];
% y = [0.5 2.5];
% im = imagesc(x,y,Vp);
% colorbar;
% pause(0.1);
 end

 % Populate cells around edge of map to model boundary
 V(2:(N+1),1) = V(2:(N+1),2);
 V(2:(N+1),N+2) = V(2:(N+1),N+1);
 V(1,2:(N+1)) = V(2,2:(N+1));
 V(N+2,2:(N+1)) = V(N+1,2:(N+1));

 sumV = sum(sum(V));
 del = abs(sumVold-sumV)
 sumVold = sumV;
end
Vp = V(2:N+1,2:N+1);
policy = pol(2:N+1,2:N+1);
figure(2); clf;
axis([0 Np+1 0 Np+1]);
axis('square');
hold on;
x = [0.5 N+0.5];
y = [0.5 N+0.5];
im = imagesc(x,y,Vp');
colorbar;
for i = 1:2:N
 for j = 1:2:N
 %if goal(i+1,j+1) == 1
 if policy(i,j)== 0
 else
 draw_arrow(i,j,1.5,-(policy(i,j)-1)*pi/2);
 end
 end
end
hold off;
figure(4); clf;
b = bar3(Vp);
colorbar
for k = 1:length(b)
 zdata = b(k).ZData;
 b(k).CData = zdata;
 b(k).FaceColor = 'interp';
end
caxis([0 1000]);
axis([0 Np+1 0 Np+1 0 1000]);
% Hill climb to goal
xy = [28;20];
% xy = [20;20];
xy_sv = xy;
Val = 0;
while Val < 1000
 ii = xy(1);
 jj = xy(2);
 if policy(ii,jj) == 1
 xy = [ii;jj+1];
 elseif policy(ii,jj) == 2
 xy = [ii+1;jj];
 elseif policy(ii,jj) == 3
 xy = [ii;jj-1];
 elseif policy(ii,jj) == 4
 xy = [ii-1;jj];
 else
 % do nothing
 end
 Val = Vp(xy(1),xy(2));
 xy_sv = [xy_sv xy];
end
figure(2); hold on;
plot(xy_sv(1,:),xy_sv(2,:),'r-');
hold off;