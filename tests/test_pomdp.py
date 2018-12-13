import numpy as np

from best.models.pomdp import POMDP, POMDPNetwork

def test_evolve():
  '''test non-deterministic connection'''
  T0 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]])
  T1 = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])

  mdp1 = POMDP([T0, T1], input_names=['u1'], state_name='x1')
  mdp2 = POMDP([T0, T1], input_names=['u2'], state_name='x2')

  network = POMDPNetwork()
  network.add_pomdp(mdp1)

  sp, _ = network.evolve([0], (0,))
  np.testing.assert_equal(sp, [1])

  network.add_pomdp(mdp2)

  sp, _ = network.evolve([1,1], (0,1))
  np.testing.assert_equal(sp, [2, 0])

  network.add_connection(['x1'], 'u2', lambda x1: set([0, 1]))

  n0 = 0
  n2 = 0
  for i in range(1000):
    sp, _ = network.evolve([1,1], (0,))

    np.testing.assert_equal(sp[0], 2)

    if sp[1] == 0:
      n0 += 1

    if sp[1] == 2:
      n2 += 1

  np.testing.assert_equal(n0 + n2, 1000)

  np.testing.assert_array_less(abs(n0 -n2), 100)


def test_Tuz():

  T0 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0.7, 0, 0.3]])
  Z0 = np.array([[0.5, 0.5], [0, 1], [1, 0]])

  pomdp = POMDP([T0], [Z0])

  Tuz = pomdp.Tuz((0,), 0).todense()   # probability of going to s and seeing z
  Tuz_r = np.array([[0, 0, 0.5], [0, 0, 0], [0.35, 0, 0.3]])

  np.testing.assert_almost_equal(Tuz, Tuz_r)

  Tuz = pomdp.Tuz((0,), 1).todense()   # probability of going to s and seeing z
  Tuz_r = np.array([[0, 0.5, 0], [0, 1, 0], [0.35, 0, 0]])

  np.testing.assert_almost_equal(Tuz, Tuz_r)