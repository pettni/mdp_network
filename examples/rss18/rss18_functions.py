import numpy as np
from best.models.pomdp import POMDP
import matplotlib.patches as patches
import polytope as pc
from functools import partial

### GENERAL UTILITIES ###

def is_adjacent(poly, x, d):
  # return true x within distance d of poly
  return np.all(poly.A.dot(x) < poly.b + d)

def vertex_to_poly(V):
  data = pc.quickhull.quickhull(V)
  return pc.Polytope(data[0], data[1])

def plot_region(ax, poly, name, prob, color='red', alpha=0.5, hatch=False, fill=True):
  ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
  _, xc = pc.cheby_ball(poly)
  ax.text(xc[0]-0.4, xc[1]-0.43, '${}_{}$\n$p={:.2f}$'.format(name[0].upper(), name[1], prob))

### BELIEF MODELS ###

def environment_belief_model(p0, levels, name):
  # Create map belief MDP with prior p0 and qw quality of weak measurements
  if p0 == 0:
    # no dynamics
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b', 
           input_trans = lambda n: 0, output_trans = lambda s: 0)
  elif p0 == 1:
    return POMDP([np.array([1])], input_names=[name+'_u'], state_name=name+'_b',
           input_trans = lambda n: 0, output_trans = lambda s: 1)
  else:
    pm = levels[0]
    pp = levels[1]

    Tnone = np.eye(5);
    Tweak = np.array([[1, 0, 0, 0, 0], 
              [0, 1, 0, 0, 0],
              [0, 1-p0, 0, p0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])
    Tstrong = np.array([[1,      0, 0, 0, 0],
              [(1-pm), 0, 0, 0, pm],
              [(1-p0), 0, 0, 0, p0],
              [(1-pp), 0, 0, 0, pp],
              [0,      0, 0, 0, 1]])

    def output_fcn(s):
      return [0, pm, p0, pp, 1][s]

  return POMDP([Tnone, Tweak, Tstrong], input_names=[name+'_u'], state_name=name+'_b', output_trans=output_fcn)

def environment_belief_model2(p0, levels, name):

  pmm = levels[0]
  pm = levels[1]
  pp = levels[2]
  ppp = levels[3]

  if p0 == 0:
    # no dynamics
    return POMDP([np.array([1])], input_trans = lambda n: 0, output_trans = lambda s: 0)
  elif p0 == 1:
    # no dynamics
    return POMDP([np.array([1])], input_trans = lambda n: 0, output_trans = lambda s: 1)
  else:
    Tnone = np.eye(7);

    Tweak = np.array([[1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, (1-p0), 0, p0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]])

    Tstrong = np.array([[1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, (1-pm), 0, 0, 0, pm, 0],
              [0, (1-p0), 0, 0, 0, p0, 0],
              [0, (1-pp), 0, 0, 0, pp, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]])

    Texact = np.array([[1, 0, 0, 0, 0, 0, 0],
              [(1-pmm), 0, 0, 0, 0, 0, pmm],
              [(1-pm), 0, 0, 0, 0, 0, pm],
              [(1-p0), 0, 0, 0, 0, 0, p0],
              [(1-pp), 0, 0, 0, 0, 0, pp],
              [(1-ppp), 0, 0, 0, 0, 0, ppp],
              [0, 0, 0, 0, 0, 0, 1]])

    def output_fcn(s):
      return [0, pmm, pm, p0, pp, ppp, 1][s]
  
  return POMDP([Tnone, Tweak, Tstrong], input_names=[name+'_u'], state_name=name+'_b', output_trans=output_fcn)

### CONNECTIONS AND PREDICATES ###

def get_rover_env_conn(region):
  name = region[0]
  poly = region[1][0]
  def conn(rx):
    dist = 1 if name[0] == 'r' else 0
    if is_adjacent(poly, rx, dist):
      return {2}
    return {0}
  return conn

def get_copter_env_conn(region, sight):
  name = region[0]
  poly = region[1][0]
  def conn(cx):
    if is_adjacent(poly, cx[0:2], 0) and cx[2] <= 2.6:
      return {2}  # strong
    elif is_adjacent(poly, cx[0:2], sight) and cx[2] > 2.6:
      return {1}  # strong
    return {0}
  return conn

def get_predicates(regions):

  predicates = dict()

  # fail predicate
  risk_names = list(filter(lambda name: name[0] == 'r', regions.keys()))
  risk_inputs = ['r_x'] + ['{}_b'.format(name) for name in risk_names]

  def risk_predicate(names, rx, *nargs):
    conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] > 0  
             for (i, name) in enumerate(names)]
    return {any(conds)}

  predicates['fail'] = (risk_inputs, partial(risk_predicate, risk_names))

  # sample predicates
  sample_types = set([name[0] for name in regions.keys()]) - set('r')

  for reg_type in sample_types:
    type_names = list(filter(lambda name: name[0] == reg_type, regions.keys()))
    type_inputs = ['r_x'] + ['{}_b'.format(name) for name in type_names]

    # Use functools.partial to avoid using same reg_type for all predicates
    def type_predicate(names, rx, *nargs):
      conds = [is_adjacent(regions[name][0], rx, 0) and nargs[i] > 0  
               for (i, name) in enumerate(names)]
      return {any(conds)}

    ap_name = 'sample{}'.format(reg_type.upper())
    predicates[ap_name] = (type_inputs, partial(type_predicate, type_names))

  return predicates

### CONTROL POLICIES ###

class RoverPolicy:
  
  def __init__(self, ltlpol, rover_abstr):
    self.ltlpol = ltlpol
    self.rover_abstr = rover_abstr

    self.t = 0
    self.s_ab = None

  def __call__(self, x_rov, s_map, APs):
    self.ltlpol.report_aps(APs)

    s_ab = self.rover_abstr.x_to_s(x_rov)
    
    if s_ab != self.s_ab and self.s_ab != None:
      self.t +=  1
    
    self.s_ab = s_ab
    u_ab, val = self.ltlpol((s_ab,) + tuple(s_map), self.t)

    if u_ab == (0,):
      self.t += 1

    return self.rover_abstr.interface(u_ab, s_ab, x_rov), val

  def get_value(self, x, s_map):
    s_ab = self.rover_abstr.x_to_s(x)
    t_act = min(self.t, len(self.ltlpol.val)-1)
    return self.ltlpol.val[t_act][(s_ab,) + tuple(s_map) + (self.ltlpol.dfsa_state,)]

  def finished(self):
    return self.ltlpol.finished() or self.t > len(self.ltlpol.val)
  
  def reset(self):
    self.ltlpol.reset()
    self.t = 0
    self.s_ab = None    

class CopterPolicy:
  
  def __init__(self, pol_list, val_list, copter_abstr):
    self.pol_list = pol_list
    self.val_list = val_list
    self.ft = False
    self.copter_abstr = copter_abstr

    self.t = 0
    self.s_ab = None
      
  def __call__(self, x_cop, s_map):
        
    s_ab = self.copter_abstr.x_to_s(x_cop)

    if s_ab != self.s_ab and self.s_ab != None:
      self.t +=  1

    if self.t >= len(self.pol_list):
      self.ft = True
      u_ab = (0,)
      val = self.val_list[-1][(s_ab,) + tuple(s_map)]

    else:
      self.s_ab = s_ab
      val = self.val_list[self.t][(s_ab,) + tuple(s_map)]
      u_ab = (self.pol_list[self.t][0][(s_ab,) + tuple(s_map)],)  # input is 1-tuple
      if u_ab == (0,):
        # stay in cell
        self.t += 1 
    return self.copter_abstr.interface(u_ab, s_ab, x_cop), val

  def reset(self):
    self.ft = False
    self.t = 0
    self.s_ab = None
  
  def finished(self):
    return self.ft

### SIMULATION ###

def reveal_map(mapstate, measurement, realmap):
  for i in range(len(measurement)):
    if mapstate[i] in [0, 4]:
      continue
    if measurement[i] == 1:  # weak meas
      if realmap[i] == 4:
        mapstate[i] = 3
      else:
        mapstate[i] = 1
    if measurement[i] == 2:  # strong meas
      mapstate[i] = realmap[i]
  return mapstate

def small_step(x_curr, x_next):
  if np.linalg.norm(x_next - x_curr) > 0:
    return x_curr + 0.1*(x_next - x_curr)/np.linalg.norm(x_next - x_curr)
  else:
    return x_curr


def rover_env_sim(rx, regions):
  '''true/false vector for regions rover is in'''
  ret = []
  for item in regions.items():
    conn = get_rover_env_conn(item)
    ret.append(list(conn(rx))[0])
  return ret

def copter_env_sim(cx, regions, copter_sight):
  '''true/false vector for regions copter is in'''
  ret = []
  for item in regions.items():
    conn = get_copter_env_conn(item, copter_sight)
    ret.append(list(conn(cx))[0])
  return ret

def rover_aps_sim(rx, map_belief, output_names, predicates):
  complete_output = (rx,) + tuple(map_belief)
  aps = set()
  for ap, (outputs, fcn) in predicates.items():
    args = [complete_output[output_names.index(output)] for output in outputs]
    if True in fcn(*args):
      aps |= {ap}
  return aps
