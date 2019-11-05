from data_utils import *
from plot_utils import *
# import matplotlib.pyplot as plt

y_init = 0
y_final = 0.25
t_peak = 4
y_peak = 1
t_decay = 15
# t_decay = 10
alpha = 0.5
tau = 0

t, y = second_order_curve(ty_peak = (t_peak, y_peak), y_init = y_init, y_final = y_final, t_decay = t_decay, alpha = alpha, T = 50, stepsize = 0.01, tau = tau)
plot_second_order(t, y, ty_peak = (t_peak, y_peak), y_init = y_init, y_final = y_final, t_decay = t_decay, alpha = alpha, tau = tau)
