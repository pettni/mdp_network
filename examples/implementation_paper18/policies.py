class CassiePolicy:
  
  def __init__(self, ltlpol, abstraction):
    self.ltlpol = ltlpol
    self.abstraction = abstraction

    self.t = 0
    self.s_ab = None

  def __call__(self, x_rov, s_map, APs):
    self.ltlpol.report_aps(APs)

    s_ab = self.abstraction.x_to_s(x_rov)
    
    if s_ab != self.s_ab and self.s_ab != None:
      self.t +=  1
    
    self.s_ab = s_ab
    u_ab, val = self.ltlpol((s_ab,) + tuple(s_map), self.t)

    if u_ab == (0,):
      self.t += 1

    return self.abstraction.interface(u_ab, s_ab, x_rov), val

  def get_value(self, x, s_map):
    s_ab = self.abstraction.x_to_s(x)
    t_act = min(self.t, len(self.ltlpol.val)-1)
    return self.ltlpol.val[t_act][(s_ab,) + tuple(s_map) + (self.ltlpol.dfsa_state,)]

  def finished(self):
    return self.ltlpol.finished() or self.t > len(self.ltlpol.val)
  
  def reset(self):
    self.ltlpol.reset()
    self.t = 0
    self.s_ab = None  


class UAVPolicy:
  
  def __init__(self, pol, val, abstraction):
    self.pol = pol
    self.val = val
    self.abstraction = abstraction
    self.ft = False
      
  def __call__(self, x_cop, s_map):
        
    s_ab = self.abstraction.x_to_s(x_cop)
    val = self.val[(s_ab,) + tuple(s_map)]
    if val == 0:
      self.ft = True
    u_ab = (self.pol[0][(s_ab,) + tuple(s_map)],)  # input is 1-tuple

    return self.abstraction.interface(u_ab, s_ab, x_cop), val
  
  def finished(self):
    return self.ft

  def reset(self):
    self.ft = False
