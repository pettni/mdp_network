import rospy
from cyberpod_ros.msg import command
from cyberpod_ros.srv import params, paramsRequest, paramsResponse

class RobCMD:

  def __init__(self):
    self.pub = rospy.Publisher('cyberpod_cmd_throttled', command, queue_size=10)
    self.srv = rospy.ServiceProxy('cyberpod_params', params)

  def goto(self, x, y):
    msg = command([x, y, 1.])
    self.pub.publish(msg)

  def hold(self):
    msg = command([0., 0., 0.])
    self.pub.publish(msg)

  def set_run(self):
    self.srv(paramsRequest(5, 2, [0.]*20))

  def set_idle(self):
    self.srv(paramsRequest(5, 1, [0.]*20))

  def set_failure(self):
    self.srv(paramsRequest(5, 0, [0.]*20))
