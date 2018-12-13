import polytope as pc
from collections import OrderedDict
import numpy as np

# problem definition
def get_prob():

  prob = {}
  prob['cas_T'] = 20

  prob['step_margin'] = 0.01
  prob['accept_margin'] = 0.8
  prob['reject_margin'] = 0.2

  prob['formula'] = '( F sampleA ) | ( F sampleB )'

  prob['xmin'] = [0, 0]
  prob['xmax'] = [3, 3]
  prob['discretization'] = [6, 6]

  prob['cas_x0'] = np.array([0.25, 0.25])
  prob['uav_x0'] = np.array([0.25, 0.25])
  prob['uav_xT'] = np.array([0.25, 0.25])

  regs = OrderedDict()
  regs['a1'] = (pc.box2poly(np.array([[2.5, 3], [1.5, 2]])), 0.5, 'blue')
  regs['b1'] = (pc.box2poly(np.array([[1.5, 2], [2.5, 3]])), 0.9, 'green')

  prob['regs'] = regs
  prob['env_x0'] = [1, 1]

  # what to reveal
  prob['REALMAP'] = [2, 0]    # SHOULD CONTAIN ZEROS AND TWOS
  return prob