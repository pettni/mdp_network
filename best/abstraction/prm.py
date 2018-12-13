import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
import networkx as nx

import polytope as pc

from best.abstraction.gridding import Abstraction
from best.models.pomdp import POMDP

class PRM(Abstraction):

  def __init__(self, x_low, x_up, num_nodes, min_dist=0, max_dist=np.Inf, informed_samples=[], name_prefix=''):
    ''' Create a PRM graph with n_nodes in [x_low, x_up]; informed sampling in informed_samples '''
    super().__init__(x_low, x_up)

    self.G = nx.Graph()
    self.min_dist = min_dist
    self.max_dist = max_dist
    
    self.sample_nodes(num_nodes, informed_samples)
    self.abstract(name_prefix)

  @property
  def N(self):
    '''number of nodes in PRM'''
    return len(self.G)

  @property
  def M(self):
    '''maximal node degree in PRM'''
    return max(d for _, d in self.G.degree)

  @property
  def costs(self):
    # small cost for not doing anything
    costs = 0.01*np.ones((self.M, self.N))
    for n0 in self.G.nodes():
      for m in range(self.M):
        n1 = self.get_kth_neighbor(n0, m)
        if n1 != n0:
          costs[m, n0] = self.distance(self.G.nodes[n0]['xc'], self.G.nodes[n1]['xc'])
    return costs

  def interface(self, u_ab, s_ab, x):
    '''return target point for given abstract control and action'''
    sp, _ = self.mdp.evolve_observe(s_ab, u_ab)
    return self.s_to_x( sp )

  def s_to_x(self, s):
    '''center of node s'''
    return self.G.nodes[s]['xc']

  def x_to_s(self, x):
    '''closest node to point x'''
    dist_list = [self.distance(self.G.nodes[n]['xc'], x) for n in range(self.N)]
    return np.argmin(dist_list)

  def sample_nodes(self, n_nodes, informed_samples=[]):
    '''add n_nodes nodes by random or informed sampling'''
    N0 = self.N
    fail_counter = 0
    sample_counter = 0

    while self.N < N0 + n_nodes and fail_counter < 100:
      if sample_counter < len(informed_samples):
        # informed sampling in regs
        xc = informed_samples[sample_counter]
      else:
        xc = self.x_low + (self.x_up - self.x_low) * np.random.rand(self.dim).ravel()
        
      sample_counter += 1
      
      dist_list = [self.distance(self.G.nodes[n]['xc'], xc) for n in range(self.N)]
      if sample_counter < len(informed_samples) or len(dist_list) == 0 or np.min(dist_list) > self.min_dist:
        n0 = self.N
        self.G.add_node(n0, xc=xc)
        for n1 in self.G.nodes():
          d01 = self.distance(xc, self.G.nodes[n1]['xc'])
          if d01 < self.max_dist:
            self.G.add_edge(n0, n1, dist=d01)
      else:
        fail_counter += 1

    if fail_counter == 100:
      print('warning: could not place', n_nodes, 'nodes')

  def distance(self, node1, node2):
      return norm(node1-node2)

  def get_kth_neighbor(self, node1, k):
    '''get successor of kth outgoing edge from node1'''
    neigh_list = sorted(self.G[node1])
    if k < len(neigh_list):
      return neigh_list[k]
    else:
      return node1

  def abstract(self, name_prefix=''):
    ''' represent graph as MDP by treating index of neigh as action number '''
    T_list = []
    n0_list = range(self.N)
    val_list = np.ones(self.N)
    for m in range(self.M):
      n1_list = [self.get_kth_neighbor(n0, m) for n0 in n0_list]
      T_list.append(sp.coo_matrix((val_list, (n0_list, n1_list)), shape=(self.N, self.N)))
    output_trans = lambda n: self.G.nodes[n]['xc']
    self.mdp = POMDP(T_list, output_name=name_prefix + '_x', output_trans=output_trans)

  def plot(self, ax):
    '''plot the PRM graph'''
    pos = {n: self.G.nodes[n]['xc'] for n in self.G.nodes()}
    nx.draw_networkx(ax=ax, G=self.G, pos=pos)
