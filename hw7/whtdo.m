h = ones(2,2);
h(1,1) = 2;
g = ones(2,2);
g(2,2) = 3;
mult = g*h
div = g/h
divequiv = g*(inv(h))

divfront = g\h
divfrontequiv = inv(g)*h