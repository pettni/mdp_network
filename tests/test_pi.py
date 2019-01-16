import numpy as np
from best.models.pomdp import *
from best.solvers.valiter import *
from best.solvers.pi import *
from best.policies.dense import *

def test_bellman_pol():

  V = np.array([1,2,3])

  T1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  T2 = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]])

  network = POMDPNetwork([POMDP([T1, T2])])

  P_ux1 = np.array([[1,1,1], [0,0,0]])
  P_ux2 = np.array([[1,0.5,1], [0, 0.5, 0]])
  P_ux3 = np.array([[0.5,0.5,1], [0.5, 0.5, 0]])

  np.testing.assert_almost_equal(bellman_policy(network, V, P_ux1), np.array([1,2,3]))
  np.testing.assert_almost_equal(bellman_policy(network, V, P_ux2), np.array([1,2.25,3]))
  np.testing.assert_almost_equal(bellman_policy(network, V, P_ux3), np.array([1.5,2.25,3]))


def test_pi1():
  accept = np.array([0,0,1])

  T1 = np.array([[0.5, 0.5, 0], [0, 1, 0], [0, 0, 1]])
  T2 = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1]])

  network = POMDPNetwork([POMDP([T1, T2])])

  V_x, P_ux = PI(network, accept, prec=1e-7)

  np.testing.assert_almost_equal(V_x, [1, 0, 1])
  np.testing.assert_almost_equal(P_ux[1, 0], 1)   # need to select action 1 in state 0   


def test_ltl_solve():

  T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
  T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

  system = POMDP([T1, T2], state_name='x')

  network = POMDPNetwork([system])

  formula = '( ( F s1 ) & ( F s2 ) )'

  predicates = {'s1': (['x'], lambda x: set([int(x==1)])),
                's2': (['x'], lambda x: set([int(x==3)]))}

  dfsa, dfsa_init, dfsa_final = formula_to_pomdp(formula)

  network.add_pomdp(dfsa)
  for ap, (outputs, conn) in predicates.items():
    network.add_connection(outputs, ap, conn)

  Vacc = np.zeros(network.N)
  Vacc[...,list(dfsa_final)[0]] = 1

  V_x, P_ux = PI(network, Vacc)

  np.testing.assert_almost_equal(V_x[:, 0], [0.5, 0, 0, 0.5], decimal=4)
