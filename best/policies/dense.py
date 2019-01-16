
class InfiniteHorizonPolicy(object):

  def __init__(self, P_xu, N, M):
    self.P_xu = P_xu


class LTL_Policy(object):
  """control policy"""
  def __init__(self, proplist, dfsa_Tlist, dfsa_init, dfsa_final, val, pol):
    '''create a control policy object'''
    self.proplist = proplist
    self.dfsa_Tlist = dfsa_Tlist
    self.dfsa_init = dfsa_init
    self.dfsa_final = dfsa_final
    self.val = val
    self.pol = pol

    self.dfsa_state = self.dfsa_init

  def reset(self):
    '''reset controller'''
    self.dfsa_state = self.dfsa_init

  def report_aps(self, aps):
    '''report atomic propositions to update internal controller state'''
    dfsa_action = tuple(int(ap in aps) for ap in self.proplist)
    row = self.dfsa_Tlist[dfsa_action].getrow(self.dfsa_state)
    self.dfsa_state = row.indices[0]

  def __call__(self, syst_state, t=0):
    '''get input from policy'''
    if t >= len(self.val)-1:
      print('Warning: t={} larger than horizon {}. Setting t={}'.format(t, len(self.val)-1, len(self.val)-2))
      t = len(self.val)-2
    joint_state = tuple(syst_state) + (self.dfsa_state,)

    u = tuple(self.pol[t][m][joint_state] for m in range(len(self.pol[t])))

    return u, self.val[t][joint_state]

  def finished(self):
    '''check if policy reached target'''
    return self.dfsa_state in self.dfsa_final
