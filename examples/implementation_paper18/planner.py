from functools import partial
from collections import OrderedDict

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import polytope as pc

from best.models.pomdp import POMDP, POMDPNetwork
from best.solvers.valiter import *
from best.abstraction.gridding import Grid
from best.abstraction.prm import PRM

from prob_simple import get_prob
from policies import *


def plan_mission(prob):

    rob_abstr = Grid(prob['xmin'], prob['xmax'], prob['discretization'], name_prefix='c')
    env_list = [environment_belief_model(info[1], name) for (name, info) in prob['regs'].items()]

    # Construct rob-env network
    rob_env_network = POMDPNetwork([rob_abstr.pomdp] + env_list)
    for item in prob['regs'].items():
        rob_env_network.add_connection(['c_x'], '{}_u'.format(item[0]), get_rob_env_conn(item))

    # solve rob LTL problem
    predicates = get_predicates(prob['regs'])
    rob_ltlpol = solve_ltl_cosafe(rob_env_network, prob['formula'], predicates,
                                  horizon=prob['cas_T'], delta=prob['step_margin'], verbose=False)

    return CassiePolicy(rob_ltlpol, rob_abstr)

def plan_exploration(prob, rob_policy):

    rob_abstr = rob_policy.abstraction
    rob_ltlpol = rob_policy.ltlpol

    informed_samples = [r.chebXc for r,_,_ in prob['regs'].values()] + [prob['uav_x0'], prob['uav_xT']]
    uav_prm = PRM(prob['xmin'], prob['xmax'], num_nodes=40, min_dist=2, max_dist=5, 
                  informed_samples=informed_samples, name_prefix='u')

    env_list = [environment_belief_model(info[1], name) for (name, info) in prob['regs'].items()]

    # Construct uav-env network
    uav_env_network = POMDPNetwork([uav_prm.mdp] + env_list)
    for item in prob['regs'].items():
        uav_env_network.add_connection(['u_x'], '{}_u'.format(item[0]), get_uav_env_conn(item))

    # solve uav exploration problem
    idx = np.logical_or(rob_ltlpol.val[0][rob_abstr.x_to_s(prob['cas_x0']), ..., rob_ltlpol.dfsa_init] > prob['accept_margin'],
                        rob_ltlpol.val[0][rob_abstr.x_to_s(prob['cas_x0']), ..., rob_ltlpol.dfsa_init] < prob['reject_margin'])

    target = np.zeros(uav_env_network.N)
    target[uav_prm.x_to_s(prob['uav_xT'])][idx] = 1

    costs = uav_prm.costs.reshape(uav_prm.costs.shape + (1,)*(1+len(uav_env_network.N) - 2))


    val_uav, pol_uav = solve_min_cost(uav_env_network, costs, target, M=500, verbose=False)

    return UAVPolicy(pol_uav, val_uav, uav_prm)

def plot_problem(prob, ax):
    ax.set_xlim(prob['xmin'][0], prob['xmax'][0])
    ax.set_ylim(prob['xmin'][1], prob['xmax'][1])
    for i, (name, info) in enumerate(prob['regs'].items()):
        plot_region(ax, info[0], name, info[1], info[2], alpha=0.5, hatch=False, fill=True if prob['REALMAP'][i] or info[1] == 1 else False)
        
def plot_value_rob(rob_policy, prob):
    def my_value(x, mapstate):    
        val = rob_policy.get_value(x, tuple(mapstate))
        return val

    def my_init_value(x, y):
        return my_value(np.array([x, y]), prob['env_x0'])

    x_low = rob_policy.abstraction.x_low
    x_up = rob_policy.abstraction.x_up

    xx, yy = np.meshgrid(np.arange(x_low[0]+0.01, x_up[0]-0.01, 0.1), 
                        np.arange(x_low[1]+0.01, x_up[1]-0.01, 0.1))

    vals = np.vectorize(my_init_value)(xx, yy)

    plt.pcolor(xx, yy, vals, vmin=0, vmax=1)
    plt.colorbar()

    print ('success probability: {}'.format(my_value(prob['cas_x0'], prob['env_x0'])))

def plot_value_uav(uav_policy, prob):
    map_init = prob['env_x0']

    def uav_value(x, mapstate):  
        scop = uav_policy.abstraction.x_to_s(x)
        return uav_policy.val[(scop,) + tuple(mapstate)]

    def my_init_cvalue(x, y):
        return uav_value(np.array([x, y]), map_init)

    x_low = uav_policy.abstraction.x_low
    x_up = uav_policy.abstraction.x_up

    xx, yy = np.meshgrid(np.arange(x_low[0]+0.01, x_up[0]-0.01, 0.1), 
                        np.arange(x_low[1]+0.01, x_up[1]-0.01, 0.1))
    vals = np.vectorize(my_init_cvalue)(xx, yy)

    plt.pcolor(xx, yy, vals)
    plt.colorbar()

    print ('success value: {}'.format(uav_value(prob['uav_x0'], map_init)))

def is_adjacent(poly, x, d):
  # return true x within distance d of poly
  return np.all(poly.A.dot(x) < poly.b + d)

def plot_region(ax, poly, name, prob, color='red', alpha=0.5, hatch=False, fill=True):
  ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
  _, xc = pc.cheby_ball(poly)
  ax.text(xc[0], xc[1], '${}_{}$\n$p={:.2f}$'.format(name[0].upper(), name[1], prob))

def get_rob_env_conn(region):
  name = region[0]
  poly = region[1][0]
  def conn(rx):
    dist = 0.6 if name[0] == 'r' else 0
    if is_adjacent(poly, rx, dist):
      return {1}
    return {0}
  return conn

def get_uav_env_conn(region):
  poly = region[1][0]
  def conn(cx):
    if is_adjacent(poly, cx, 0):
      return {1} 
    return {0}
  return conn

def environment_belief_model(p0, name):
  # Create map belief MDP with prior p0
  if p0 == 0:
    # no dynamics
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b', 
           input_trans = lambda n: 0, output_trans = lambda s: 0)
  if p0 == 1:
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b',
           input_trans = lambda n: 0, output_trans = lambda s: 1)
  
  Tnone = np.eye(3);
  Tmeas = np.array([[1.  , 0,  0],
                    [1-p0, 0, p0],
                    [0,    0,  1]]);

  return POMDP([Tnone, Tmeas], input_names=[name+'_u'], state_name=name+'_b',
               output_trans=lambda s: [0, p0, 1][s])

def get_predicates(regions):

  predicates = dict()

  if any([name[0] == 'r' for name in regions.keys()]):
    # fail predicate
    risk_names = list(filter(lambda name: name[0] == 'r', regions.keys()))
    risk_inputs = ['c_x'] + ['{}_b'.format(name) for name in risk_names]

    def risk_predicate(names, rx, *nargs):
      conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] > 0  
               for (i, name) in enumerate(names)]
      return {any(conds)}

    predicates['fail'] = (risk_inputs, partial(risk_predicate, risk_names))

  # sample predicates
  sample_types = set([name[0] for name in regions.keys()]) - set('r')

  for reg_type in sample_types:
    type_names = list(filter(lambda name: name[0] == reg_type, regions.keys()))
    type_inputs = ['c_x'] + ['{}_b'.format(name) for name in type_names]

    # Use functools.partial to avoid using same reg_type for all predicates
    def type_predicate(names, rx, *nargs):
      conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] == 1  
               for (i, name) in enumerate(names)]
      return {any(conds)}

    ap_name = 'sample{}'.format(reg_type.upper())
    predicates[ap_name] = (type_inputs, partial(type_predicate, type_names))

  return predicates

def main():

  np.random.seed(4)

  prob = get_prob()
  plot_problem(prob)

  cas_policy = plan_mission(prob)

  uav_policy = plan_exploration(prob, cas_policy)

  plot_value_rob(cas_policy, prob)
  plt.show()

  plot_value_uav(uav_policy, prob)
  plt.show()

if __name__ == '__main__':
  main()

