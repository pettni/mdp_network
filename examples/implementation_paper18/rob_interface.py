import struct

import rospy
from ambercortex_ros.msg import cmd

class RobCMD:

  def __init__(self):
    self.pub = rospy.Publisher('cyberpod/cmd', cmd, queue_size=10)

  def goto(self, x, y):
    byte_content = list(struct.pack('ff', x, y))
    chksum = sum(byte_content) % 256
    msg = cmd(data=byte_content, chksum=chksum)
    self.pub.publish(msg)

  def hold(self):
    msg = cmd([0., 0., 0.])
    self.pub.publish(msg)