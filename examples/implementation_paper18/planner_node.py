#!/usr/bin/env python

import struct
import numpy as np
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

from planner import *
from policies import *
from rob_interface import RobCMD
from uav_interface import UAVCMD

if False:
  from prob_simple import get_prob
else:
  from prob_cast import get_prob

SIM = True

UDP_PORT = 1560

MATLAB_QUADROTOR_PATH = r'/mnt/c/Users/petter/coding/quadrotor/lib'

if SIM:
  UDP_IP = '127.0.0.1'
  UAV_ALTITUDE = 1.5  # m
  UAV_SPEED = 0.3     # m/s
  UAV_POSE_TOPIC = '/MATLAB_UAV'
  ROB_POSE_TOPIC = '/MATLAB_ROB'
  LAND_CUTOFF = 0.94
else:
  UDP_IP = '192.168.0.4'
  UAV_ALTITUDE = 1.5  # m
  UAV_SPEED = 0.3     # m/s
  UAV_POSE_TOPIC = '/vrpn_client_node/AMBERUAV/pose'
  ROB_POSE_TOPIC = '/vrpn_client_node/AMBERPOD/pose'
  LAND_CUTOFF = 1.3

prob = get_prob()

# Fake function 1: reveal map
def reveal_map_uav(mapstate, uav_pos):
  ret = mapstate
  for i, (name, item) in enumerate(prob['regs'].items()):
    if is_adjacent(item[0], uav_pos[0:2], 0):
      ret[i] = prob['REALMAP'][i]

  # print("mapstate changed to", ret)
  return ret

def is_landed(uav_pose, rob_pose):
  return uav_pose[2] < LAND_CUTOFF

# Fake function 2: reveal map
def reveal_map_rob(mapstate, rob_pos):
  ret = mapstate
  for i, (name, item) in enumerate(prob['regs'].items()):
    if is_adjacent(item[0], rob_pos[0:2], 0.5):
      ret[i] = prob['REALMAP'][i]

  # print("mapstate changed to", ret)
  return ret

# reveal APs
def map_i_output(i, mapstate_i):
    p = list(prob['regs'].items())[i][1][1]
    if p == 1 or p == 0:
        return p
    return [0, p, 1][mapstate_i]

def map_output(mapstate):
    return tuple(map_i_output(i, mapstate[i]) for i in range(len(prob['regs'])))

def robot_aps(rx, mapstate):
  predicates = get_predicates(prob['regs'])
  complete_output = (rx,) + map_output(mapstate)
  output_names = ['c_x'] + [key + '_b' for key in prob['regs'].keys()] 
  aps = set()
  for ap, (outputs, fcn) in predicates.items():
    args = [complete_output[output_names.index(output)] for output in outputs]
    if True in fcn(*args):
      aps |= {ap}
  return aps

class Planner(object):

  def __init__(self, rob_cmd, uav_cmd):
    np.random.seed(4)

    self.rob_cmd = rob_cmd
    self.uav_cmd = uav_cmd

    self.uav_pos = None
    self.rob_pos = None

    self.uav_pol = None
    self.rob_pol = None

    self.mapstate = prob['env_x0']         # state of map exploration (0 false, 1 unknown, 2 positive)

    self.change_state('plan_mission')   
    self.uavstate = 'landed'  
    self.mission_proba = 0.

    self.pub_prob = rospy.Publisher('/probability', Float32, queue_size=10)
    self.pub_dist = rospy.Publisher('/expected_distance', Float32, queue_size=10)

  def uav_callback(self, msg):
    self.uav_pos = np.array([msg.pose.position.x,
                             msg.pose.position.y, 
                             msg.pose.position.z])
    self.mapstate = reveal_map_uav(self.mapstate, self.uav_pos)   # reveal hidden things..    

  def rob_callback(self, msg):

    siny_cosp = 2.0 * (msg.pose.orientation.w * msg.pose.orientation.z 
                       + msg.pose.orientation.x * msg.pose.orientation.y);
    cosy_cosp = 1.0 - 2.0 * (msg.pose.orientation.y * msg.pose.orientation.y
                             + msg.pose.orientation.z * msg.pose.orientation.z); 
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    self.rob_pos = np.array([msg.pose.position.x, msg.pose.position.y, yaw])  
    self.mapstate = reveal_map_rob(self.mapstate, self.rob_pos)   # reveal hidden things..    

  def change_state(self, newstate):
    
    # entry actions are defined here
    print('switching to', newstate)

    if newstate == 'plan_mission':
      self.rob_pol = None
      self.state = 'plan_mission'

    if newstate == 'execute_mission':
      self.state = 'execute_mission'

    if newstate == 'plan_exploration':
      self.uav_pol = None
      self.state = 'plan_exploration'

    if newstate == 'explore':
      if SIM:
        self.rob_cmd.goto(self.rob_pos[0], self.rob_pos[1])  # stop here
      else:
        self.rob_cmd.hold()
      self.state = 'explore'

    if newstate == 'done':
      self.state = 'done'

  def step(self):
    
    # STATE MACHINE WITH FIVE STATES
    #
    # plan_mission, execute_mission, plan_exploration, explore, done

    if self.state == 'plan_mission':

      # during
      if self.rob_pos is not None:
        print("planning mission..")
        prob['cas_x0'] = np.array(self.rob_pos[0:2])
        self.rob_pol = plan_mission(prob)
      else:
        print("waiting for robot position data")

      # exit
      if self.rob_pol is not None:
        self.change_state('execute_mission')
    
    elif self.state == "execute_mission":
      # during
      aps = robot_aps( (self.rob_pos[0], self.rob_pos[1]), self.mapstate )
      if len(aps):
        print("reported APs", aps)
      target, val = self.rob_pol(self.rob_pos[0:2], self.mapstate, aps)
      self.pub_prob.publish(val)
      with open('proba.txt', 'a') as f:
        f.write(str(rospy.get_rostime()) + " " + str(val) + '\n') 
      if val > prob['accept_margin']:
        print("segway going to", target)
        self.rob_cmd.goto(target[0], target[1])

      # exit
      if self.rob_pol.finished():
        self.change_state('done')
      if not(val < prob['reject_margin'] or val > prob['accept_margin']):
        self.change_state('plan_exploration')
      if val == 0:
        self.change_state('done')

    elif self.state == 'plan_exploration':
      
      # during
      if self.rob_pos is not None:
        print("planning exploration..")
        prob['cas_x0'] = np.array(self.rob_pos[0:2])
        prob['uav_x0'] = np.array(self.rob_pos[0:2])
        prob['uav_xT'] = np.array(self.rob_pos[0:2])
        self.uav_pol = plan_exploration(prob, self.rob_pol) 
      else:
        print("waiting for position data")

      # exit
      if self.uav_pol is not None:
        self.change_state('explore')

    elif self.state == 'explore':

      # during
      if self.uavstate == 'flying' and self.uav_pol.finished():
        print("sending land")
        self.uav_cmd.land_on_platform(self.rob_pos)
        
        if is_landed(self.uav_pos, self.rob_pos):
          time.sleep(5)
          self.uavstate = 'landed'
 
      elif self.uavstate == 'flying':
        target, exp_dist = self.uav_pol(self.uav_pos[0:2], self.mapstate)
        self.pub_dist.publish(exp_dist)
        with open('exp_dist.txt', 'a') as f:
          f.write(str(rospy.get_rostime()) + " " + str(exp_dist) + '\n') 

        _, rob_val = self.rob_pol(self.rob_pos[0:2], self.mapstate, {})
        with open('proba.txt', 'a') as f:
          f.write(str(rospy.get_rostime()) + " " + str(rob_val) + '\n') 

        self.pub_prob.publish(rob_val)
        print("UAV going to", target)
        self.uav_cmd.goto(target[0], target[1], UAV_ALTITUDE, UAV_SPEED)

      elif self.uavstate == 'landed':
        print("sending takeoff in 4s")
        time.sleep(4)
        print("sending takeoff")
        self.uav_cmd.takeoff()
        time.sleep(0.5)
        self.uavstate = 'flying'

      # exit
      if self.uavstate == 'landed' and self.uav_pol.finished():
        self.change_state('execute_mission')

    elif self.state == 'done':
      pass

    else:
      raise Exception("unknown state")


def main():
  plot_problem(prob)

  rob_cmd = RobCMD()
  uav_cmd = UAVCMD(UDP_IP, UDP_PORT)

  planner = Planner(rob_cmd, uav_cmd)

  rospy.Subscriber(UAV_POSE_TOPIC, PoseStamped, planner.uav_callback)
  rospy.Subscriber(ROB_POSE_TOPIC, PoseStamped, planner.rob_callback)
  
  rospy.init_node('best_planner', anonymous=True)
  
  rate = rospy.Rate(0.25)

  while not (rospy.is_shutdown() or planner.state == 'done'):
    planner.step()
    rate.sleep()

  rospy.signal_shutdown("planning ended") 
  return 0

if __name__ == '__main__':
  main()