import numpy as np
import sparse
from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.occupation_lp import *

def test_occupation1():

  # nondeterministic problem, compare solutions
  network = POMDPNetwork()

  T0 = np.array([[0, 1, 0, 0], 
                 [0, 0, 0.5, 0.5],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1]]);

  network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))

  T0 = np.array([[1, 0], 
                 [0, 1]]);
  T1 = np.array([[0, 1], 
                 [0, 1]]);

  network.add_pomdp(POMDP([T0, T1], input_names=['l'], state_name='q'))

  network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([1]), 2: set([0]),  3: set([0])}[s])

  # Define target set
  accept = np.zeros((4,2))
  accept[:,1] = 1

  val_list, pol_list = solve_reach(network, accept)

  P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
  P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
  conn = network.connections[0][2]

  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

  reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])


  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

  reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])


  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])

  reach_prob, _ = solve_exact(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])


def test_occupation2():

  # nondeterministic problem without solution
  network = POMDPNetwork()

  T0 = np.array([[0, 1, 0, 0], 
                 [0, 0, 0.5, 0.5],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1]]);

  network.add_pomdp(POMDP([T0], input_names=['a'], state_name='s'))

  T0 = np.array([[1, 0], 
                 [0, 1]]);
  T1 = np.array([[0, 1], 
                 [0, 1]]);

  network.add_pomdp(POMDP([T0, T1], input_names=['l'], state_name='q'))

  network.add_connection(['s'], 'l', lambda s: {0: set([0]), 1: set([0, 1]), 2: set([0]),  3: set([0])}[s])

  # Define target set
  accept = np.zeros((4,2))
  accept[:,1] = 1

  val_list, pol_list = solve_reach(network, accept)

  P_asS = diagonal(get_T_uxXz(network.pomdps['s']), 2, 3)
  P_lqQ = diagonal(get_T_uxXz(network.pomdps['q']), 2, 3)
  conn = network.connections[0][2]

  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=0, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][0, 0])

  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=1, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][1, 0])

  reach_prob, _ = solve_robust(P_asS, P_lqQ, conn, s0=2, q0=0, q_target=1)
  np.testing.assert_almost_equal(reach_prob, val_list[0][2, 0])
