from best.solvers.pi import *
from best.models.pomdp import *

accept = np.array([0,0,0,0,1])

# blue action
T1 = np.array([[0.5, 0.5, 0, 0, 0], [1, 0, 0, 0, 0], 
			   [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
# red action
T2 = np.array([[0, 0.8, 0.2, 0, 0], [0, 0, 0.5, 0.5, 0],
			   [0, 0, 0, 0.5, 0.5], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

network = POMDPNetwork([POMDP([T1, T2])])

V_x, P_ux = PI(network, accept, prec=1e-7)
