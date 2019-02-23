import polytope as pc
from collections import OrderedDict
import numpy as np

# problem definition
def get_prob():

  prob = {}
  prob['cas_T'] = 21

  prob['step_margin'] = 0.0001
  prob['accept_margin'] = 0.9
  prob['reject_margin'] = 0.2

  prob['formula'] = '( ! fail U ( sampleA & F exploreB ) ) | ( ! fail U ( emptyA & sampleC & F exploreD ) )'

  prob['xmin'] = [0, 0]
  prob['xmax'] = [10, 10]
  prob['discretization'] = [10, 10]

  prob['cas_x0'] = np.array([0.25, 0.25])
  prob['uav_x0'] = np.array([0.25, 0.25])
  prob['uav_xT'] = np.array([0.25, 0.25])

  regs = OrderedDict()
  regs['a1'] = (pc.box2poly(np.array([[6, 7], [2, 3]])), 0.6, 'green')
  regs['b1'] = (pc.box2poly(np.array([[0, 1], [7, 8]])), 0.7, 'blue')
  regs['c1'] = (pc.box2poly(np.array([[8, 9], [9, 10]])), 0.5, 'red')
  regs['d1'] = (pc.box2poly(np.array([[4, 5], [8, 9]])), 0.5, 'orange')

  regs['r1'] = (pc.box2poly(np.array([[2, 3], [5, 9]])), 0.5, 'red')
  regs['r2'] = (pc.box2poly(np.array([[5, 8], [6, 7]])), 0.5, 'red')

  prob['regs'] = regs
  prob['env_x0'] = [0 if  reg[1] in [0,1] else 1 for reg in regs.values()]

  # what to reveal
  prob['REALMAP'] = [0, 0, 0, 0, 0, 0]
  return prob