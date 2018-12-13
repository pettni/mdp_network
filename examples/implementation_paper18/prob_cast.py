import polytope as pc
from collections import OrderedDict
import numpy as np

# problem definition
def get_prob():

  prob = {}
  prob['cas_T'] = 30

  prob['step_margin'] = 0.0001
  prob['accept_margin'] = 0.9
  prob['reject_margin'] = 0.2

  prob['formula'] = '( ( ! fail U sampleA ) | ( ! fail U sampleB ) ) | ( ! fail U ( sampleC ) )'

  prob['xmin'] = [0, 0]
  prob['xmax'] = [4, 8]
  prob['discretization'] = [8, 16]

  prob['cas_x0'] = np.array([0.25, 0.75])
  prob['uav_x0'] = np.array([0.25, 0.75])
  prob['uav_xT'] = np.array([0.25, 0.75])

  regs = OrderedDict()
  regs['a1'] = (pc.box2poly(np.array([[2, 2.5], [2, 2.5]])), 0.9, 'green')
  regs['b1'] = (pc.box2poly(np.array([[3.5, 4], [7.5, 8]])), 0.7, 'blue')
  regs['c1'] = (pc.box2poly(np.array([[0.5, 1], [7, 7.5]])), 0.5, 'blue')

  regs['r1'] = (pc.box2poly(np.array([[2.5, 3], [6.5, 8]])), 0.5, 'red')
  regs['r2'] = (pc.box2poly(np.array([[3, 4], [4, 4.5]])), 0.4, 'red')
  regs['r3'] = (pc.box2poly(np.array([[2.5, 3], [4, 6.5]])), 1, 'red')

  prob['regs'] = regs
  prob['env_x0'] = [0 if  reg[1] in [0,1] else 1 for reg in regs.values()]

  # what to reveal
  prob['REALMAP'] = [0, 0, 2, 0, 2, 0]
  return prob