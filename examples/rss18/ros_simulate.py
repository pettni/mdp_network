import numpy as np
import rospy
from env_exploration.srv import *

import matplotlib.pyplot as plt

rospy.wait_for_service('copter_waypoints')

cop_waypts = rospy.ServiceProxy('copter_waypoints', Waypoint)
rov_waypts = rospy.ServiceProxy('rover_waypoints', Waypoint)

cop_init = np.array([-4.5, -1.5, 2])
rov_init = np.array([-4.5, -0.5])

# reset all
rov_req = WaypointRequest()
cop_req = WaypointRequest()
rov_req.reset_map = True
cop_req.reset_map = True
rov_req.current.x = rov_init[0]
rov_req.current.y = rov_init[1]

rov_waypts(rov_req)
cop_waypts(cop_req)

# initialize variables
rover_finished = False
copter_finished = False
value = 1.
map_belief = None

x_copter = cop_init.reshape((1,3))
x_rover = rov_init.reshape((1,2))
vv = np.array([])

# rover
while not rover_finished and value > 0.85:

  x_curr = x_rover[-1, :].flatten()
  rov_req.current.x = x_curr[0]
  rov_req.current.y = x_curr[1]

  rov_req.reset_map = False

  rov_res = rov_waypts(rov_req)
  
  x_next = np.zeros(2)
  x_next[0] = rov_res.target.x
  x_next[1] = rov_res.target.y

  value = rov_res.value
  rover_finished = rov_res.finished
  map_belief = rov_res.map_belief

  if np.linalg.norm(x_next - x_curr) > 0:
      x_del = 0.1*(x_next - x_curr)/np.linalg.norm(x_next - x_curr)
  else:
      x_del = 0
  x_rover = np.vstack([x_rover, x_curr + x_del])

  vv = np.hstack([vv, value])

print 'before exploration: value {}, map belief: {}'.format(value, map_belief)

# copter
while not copter_finished:

  x_curr = x_copter[-1, :].flatten()

  cop_req.current.x = x_curr[0]
  cop_req.current.y = x_curr[1]
  cop_req.current.z = x_curr[2]

  cop_req.reset_map = False

  print cop_req
  cop_res = cop_waypts(cop_req)
  print cop_res
  x_next = np.zeros(3)
  x_next[0] = cop_res.target.x
  x_next[1] = cop_res.target.y
  x_next[2] = cop_res.target.z

  value = cop_res.value

  copter_finished = cop_res.finished
  map_belief = cop_res.map_belief

  if np.linalg.norm(x_next - x_curr) > 0:
      x_del = 0.1*(x_next - x_curr)/np.linalg.norm(x_next - x_curr)
  else:
      x_del = 0

  x_copter = np.vstack([x_copter, x_curr + x_del])
  vv = np.hstack([vv, value])

print 'after exploration: value {}, map belief: {}'.format(value, map_belief)

# rover again
while not rover_finished and value > 0.85:

  x_curr = x_rover[-1, :].flatten()
  rov_req.current.x = x_curr[0]
  rov_req.current.y = x_curr[1]

  rov_req.reset_map = False

  rov_res = rov_waypts(rov_req)
  
  x_next = np.zeros(2)
  x_next[0] = rov_res.target.x
  x_next[1] = rov_res.target.y

  value = rov_res.value
  rover_finished = rov_res.finished
  map_belief = rov_res.map_belief

  if np.linalg.norm(x_next - x_curr) > 0:
      x_del = 0.1*(x_next - x_curr)/np.linalg.norm(x_next - x_curr)
  else:
      x_del = 0
  x_rover = np.vstack([x_rover, x_curr + x_del])
  vv = np.hstack([vv, value])


fig = plt.figure()
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)

ax.plot(x_copter[:, 0], x_copter[:, 1], color='blue', linewidth=2)
ax.plot(x_rover[:, 0], x_rover[:, 1], color='red', linewidth=2)

fig2 = plt.figure()
ax2 = plt.Axes(fig2, [0.,0.,1.,1.])
fig2.add_axes(ax2)
ax2.plot(vv)

plt.show()