import numpy as np

from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.valiter import *

def test_connection():
  T0 = np.array([[0, 1], [1, 0]])
  T1 = np.array([[1, 0], [0, 1]])
  mdp1 = POMDP([T0, T1], input_names=['u'], state_name='zout')

  T0 = np.array([[1, 0], [1, 0]])
  T1 = np.array([[0, 1], [0, 1]])

  mdp2 = POMDP([T0, T1], input_names=['zin'], state_name='y')

  network = POMDPNetwork()
  network.add_pomdp(mdp1)
  network.add_pomdp(mdp2)

  network.add_connection(['zout'], 'zin', lambda n1: set([n1]))

  V1 = np.zeros([2,2])
  V1[0,1] = 1
  vals1, _ = solve_reach(network, V1)
  np.testing.assert_almost_equal(vals1[0], [[0, 1], [0, 0]])

  V2 = np.zeros([2,2])
  V2[0,0] = 1
  vals2, _ = solve_reach(network, V2)
  np.testing.assert_almost_equal(vals2[0], [[1, 1], [1, 1]])

def test_reach():
  T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
  mdp = POMDP([T0])
  
  network = POMDPNetwork()
  network.add_pomdp(mdp)

  V_acc = np.zeros([3])
  V_acc[2] = 1

  v_list, _ = solve_reach(network, V_acc)

  np.testing.assert_almost_equal(v_list[0], [0.5, 0, 1], decimal=4)

def test_mdp_dfsa():
  def output(n1):
    if n1 == 2:
      return 1
    else:
      return 0

  T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
  mdp = POMDP([T0], output_trans=output, input_names=['mdp_in'], state_name='mdp_out')

  T1 = np.array([[1, 0], [0, 1]])
  T2 = np.array([[0, 1], [0, 1]])
  fsa = POMDP({(0,): T1, (1,): T2}, input_names=['fsa_in'], state_name='fsa_out')

  network = POMDPNetwork()
  network.add_pomdp(mdp)
  network.add_pomdp(fsa)

  network.add_connection(['mdp_out'], 'fsa_in', lambda n1: set([n1]))

  V_acc = np.zeros([3,2])
  V_acc[:,1] = 1

  v_list, _ = solve_reach(network, V_acc)

  np.testing.assert_almost_equal(v_list[0], 
                   [[0.5, 1], [0, 1], [1, 1]],
                   decimal=4)

def test_mdp_dfsa_nondet():

  def connection(n1):
    if n1 == 2:
      return set([1])
    elif n1 == 1:
      return set([1, 0])
    else:
      return set([0])

  T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
  mdp = POMDP([T0], input_names=['mdp_in'], state_name='mdp_out')

  T1 = np.array([[1, 0], [0, 1]])
  T2 = np.array([[0, 1], [0, 1]])
  fsa = POMDP({(0,): T1, (1,): T2}, input_names=['fsa_in'], state_name='fsa_out')

  network = POMDPNetwork()
  network.add_pomdp(mdp)
  network.add_pomdp(fsa)

  network.add_connection(['mdp_out'], 'fsa_in', connection)

  V_acc = np.zeros([3,2])
  V_acc[:,1] = 1

  v_list, _ = solve_reach(network, V_acc)

  np.testing.assert_almost_equal(v_list[0],
                   [[0.5, 1], [0, 1], [1, 1]],
                   decimal=4)

def test_reach_finitetime():

  T0 = np.array([[0.9, 0, 0.1], [0, 1, 0], [0, 0, 1]])
  T1 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])

  mdp = POMDP([T0, T1])

  network = POMDPNetwork()
  network.add_pomdp(mdp)

  V_acc = np.zeros([3])
  V_acc[2] = 1

  v_list, _ = solve_reach(network, V_acc, horizon=3)

  np.testing.assert_almost_equal(v_list[0][0], 0.1 + 0.9*0.1 + 0.9**2*0.5)
  np.testing.assert_almost_equal(v_list[1][0], 0.1 + 0.9*0.5)
  np.testing.assert_almost_equal(v_list[2][0], 0.5)

def test_ltl_synth():

  T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
  T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

  system = POMDP([T1, T2], state_name='x')

  network = POMDPNetwork([system])

  formula = '( ( F s1 ) & ( F s2 ) )'

  preds = {'s1': (['x'], lambda x: set([int(x==1)])),
           's2': (['x'], lambda x: set([int(x==3)]))}

  pol = solve_ltl_cosafe(network, formula, preds)

  np.testing.assert_almost_equal(pol.val[0][:,0], [0.5, 0, 0, 0.5], decimal=4)

def test_reach_constrained():

  T0 = np.array([[0.9, 0, 0.1], [0, 1, 0], [0, 0, 1]])
  T1 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])

  mdp = POMDP([T0, T1])

  network = POMDPNetwork()
  network.add_pomdp(mdp)

  V_acc = np.zeros([3])
  V_acc[2] = 1

  V_con = 0.5*np.ones([3])

  v_list, _ = solve_reach_constrained(network, V_acc, [(V_con, 0)], horizon=3)

  np.testing.assert_almost_equal(v_list[0][0], 0.1 + 0.9*0.1 + 0.9**2*0.5)
  np.testing.assert_almost_equal(v_list[1][0], 0.1 + 0.9*0.5)
  np.testing.assert_almost_equal(v_list[2][0], 0.5)


def test_reach_constrained2():

  T0 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])

  mdp = POMDP([T0])

  network = POMDPNetwork()
  network.add_pomdp(mdp)

  V_acc = np.array([0,1,1])
  V_con = np.array([0,0,1])

  v_list, _ = solve_reach_constrained(network, V_acc, [(V_con, 0)], horizon=3)
  np.testing.assert_almost_equal(v_list[0], [1, 1, 1])

  v_list, _ = solve_reach_constrained(network, V_acc, [(V_con, 0.49)], horizon=3)
  np.testing.assert_almost_equal(v_list[0], [0.5, 0, 1])

  v_list, _ = solve_reach_constrained(network, V_acc, [(V_con, 0.51)], horizon=3)
  np.testing.assert_almost_equal(v_list[0], [0, 0, 1])