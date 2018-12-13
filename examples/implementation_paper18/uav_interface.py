import socket
import struct

import numpy as np
import matlab.engine
from geometry_msgs.msg import Pose2D

MATLAB_QUADROTOR_PATH = r'/mnt/c/Users/petter/coding/quadrotor/lib'

def fit_poly_matlab(eng, t_ivals, xyz_ivals):
  t_ivals_m = matlab.double(list(t_ivals))
  x_ivals_m = matlab.double(list(xyz_ivals[0,:]))
  y_ivals_m = matlab.double(list(xyz_ivals[1,:]))
  z_ivals_m = matlab.double(list(xyz_ivals[2,:]))

  d_m = matlab.double([10])
  r_m = matlab.double([4])

  x_res_m = eng.optimize1d(t_ivals_m, x_ivals_m, d_m, r_m, nargout=2)
  y_res_m = eng.optimize1d(t_ivals_m, y_ivals_m, d_m, r_m, nargout=2)
  z_res_m = eng.optimize1d(t_ivals_m, z_ivals_m, d_m, r_m, nargout=2)

  udp_message_m = eng.pack_udp_message(t_ivals_m, x_res_m[0], y_res_m[0], z_res_m[0], nargout=1)
  udp_message = [udp_message_m[0][i] for i in range(udp_message_m.size[1])]
  cmd_b = struct.pack('{}B'.format(len(udp_message)), *udp_message)

  return cmd_b

def compute_ckhsum(msg):
  '''compute checksum for bytearray'''
  chksum = struct.pack('B', sum(msg) % 256)
  return chksum

def rob_to_platform(rob_pose):
  x = rob_pose[0] - 0.07 * np.cos(rob_pose[2])
  y = rob_pose[1] - 0.07 * np.sin(rob_pose[2])
  z = 0.93
  return x,y,z

class UAVCMD:

  def __init__(self, IP, PORT):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.cmd_nr = 0
    self.IP = IP
    self.PORT = PORT
    self.matlab_eng = matlab.engine.start_matlab()
    self.matlab_eng.addpath(MATLAB_QUADROTOR_PATH, nargout=0)


  def send_command(self, cmd_type, cmd_b=b''):
    # send UDP type-cmd_type command to UAV

    # syntax :  FCTK+L+cmd
    #
    #  F: uint8 F
    #  C: uint8 C
    #  T: uint8 'cmd_type'                      1   byte
    #  K: uint8 counter                         1   byte
    #  L: uint32 number of bytes in cmd         4   byte
    #  cmd_b: data                              4xL byte
    #  00: two empty padding bytes              2   byte
    #  chksum: checksum of everything after F   1   byte

    if len(cmd_type) > 1:
      raise Exception('type must be char')

    udp_header = struct.pack('B', ord('C'));

    udp_content = struct.pack('BB', ord(cmd_type), self.cmd_nr) \
                  + struct.pack('I', len(cmd_b)) + cmd_b + b'\x00\x00'
    udp_chksum = compute_ckhsum(udp_header + udp_content)

    udp_msg = struct.pack('B', ord('F')) + udp_header + udp_content + udp_chksum
    self.sock.sendto(udp_msg, (self.IP, self.PORT));
    self.cmd_nr = (self.cmd_nr + 1) % 256

  def land_on_platform(self, rob_pose, speed = None):
    x,y,z = rob_to_platform(rob_pose)
    print("sending land at", x, y, z, "for robot pose", rob_pose)
    yaw = rob_pose[2]
    if speed is None:
      speed = 0.1
    self.send_command('X', struct.pack('fffff', x, y, z, speed, yaw))

  def land(self):
    self.send_command('Q')

  def abort(self):
    self.send_command('A')

  def takeoff(self, speed = None):
    if speed is not None:
      self.send_command('S', struct.pack('f', speed))
    else:
      self.send_command('S')

  def trajectory(self, t_ivals, xyz_ivals):
    cmd = fit_poly_matlab(self.matlab_eng, t_ivals, xyz_ivals)
    self.send_command('F', cmd)

  def goto(self, x, y, z, speed = None):
    if speed is not None:
      self.send_command('G', struct.pack('ffff', x, y, z, speed))
    else:
      self.send_command('G', struct.pack('fff', x, y, z))
