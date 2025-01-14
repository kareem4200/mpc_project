import carla
import numpy as np
from mpc_v2 import MPC
import time
from agents.navigation.global_route_planner import GlobalRoutePlanner

client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world('Town01')
amap = world.get_map()

sampling_resolution = 5
grp = GlobalRoutePlanner(amap, sampling_resolution)

spawn_points = world.get_map().get_spawn_points()
a = carla.Location(spawn_points[50].location)
b = carla.Location(spawn_points[100].location)

w1 = grp.trace_route(a, b)
waypoints_list = []
for w in w1:
      loc = w[0].transform.location
      # print(f"X={loc.x}, Y={loc.y}, Z={loc.z}")
      waypoints_list.append([loc.x, loc.y])

      world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
      color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
      persistent_lines=True)
      
waypoints_np = np.array(waypoints_list)

blueprint_library = world.get_blueprint_library()
vehicle_blueprint = blueprint_library.filter('vehicle.*model3*')[0]
vehicle = world.spawn_actor(vehicle_blueprint, spawn_points[50])

time.sleep(2)
control = carla.VehicleControl()
time.sleep(2)

transform = vehicle.get_transform()
loc = transform.location
rot = transform.rotation

# print(waypoints_np[0])
# print(waypoints_np.shape)
# print(type(waypoints_np))

horizon = 20
dt = 0.045

done = False
initial_state = np.array([loc.x, loc.y, rot.yaw*np.pi/180, 1.0])    # (X, Y, Orientation, Velocity)
states = np.array([initial_state])
steer_arr = []
yaw_rate_arr = [0]
mpc = MPC(time_step=dt, horizon=horizon, initial_state=initial_state)

while not done:
      # print(f'Iteration: {i}')
      new_state, steer, throttle, done, yaw_rate = mpc.mpc_run(trajectory=waypoints_np)
      # print(new_state[0])
      # print(throttle[0])
      states = np.append(states, np.array([new_state]), axis=0)
      steer_arr.append(steer[0])
      yaw_rate_arr.append(yaw_rate)
      
      # print(f"Throttle: {throttle[0]}, Steer: {steer[0]}")
      # print(states[0])
      control.throttle = throttle[0]
      control.steer = steer[0]

      vehicle.apply_control(control)
      
      transform = vehicle.get_transform()
      print("current (carla): ", [transform.location.x, transform.location.y])