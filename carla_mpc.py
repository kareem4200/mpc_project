import carla
import numpy as np
from mpc_v2 import MPC
import time
from agents.navigation.global_route_planner import GlobalRoutePlanner
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2

    
client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world('Town01')

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.045
world.apply_settings(settings)

amap = world.get_map()

sampling_resolution = 5
grp = GlobalRoutePlanner(amap, sampling_resolution)

spawn_points = world.get_map().get_spawn_points()
a = carla.Location(spawn_points[50].location)
b = carla.Location(spawn_points[100].location)

w1 = grp.trace_route(a, b)
print(w1[0])
waypoints_list = []
for w in w1[1:]:
      loc = w[0].transform.location
      waypoints_list.append([loc.x, loc.y])
      world.debug.draw_point(w[0].transform.location, size=0.05, life_time=1000.0)
      
waypoints_np = np.array(waypoints_list)

blueprint_library = world.get_blueprint_library()
vehicle_blueprint = blueprint_library.filter('vehicle.*model3*')[0]

vehicle = world.spawn_actor(vehicle_blueprint, spawn_points[50])

camera_bp = blueprint_library.find("sensor.camera.rgb")

image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

camera_transform = carla.Transform(carla.Location(x=-6.0, z=3.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

sensor_data = {'rgb_image': np.zeros((image_h, image_w, 4))}
camera.listen(lambda image: cam_callback(image, sensor_data))

def cam_callback(image, data_dict):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img[:,:,3] = 255
    data_dict['rgb_image'] = img
    
control = carla.VehicleControl()

horizon = 10
dt = 0.045
done = False

mpc = MPC(time_step=dt, horizon=horizon, trajectory=waypoints_np)
i = 0

while not done:
      world.tick()
      
      cv2.imshow("RGB_Image", sensor_data['rgb_image'])
      if cv2.waitKey(1) == ord('q'):
            break
      
      # if i == 1:
      #       time.sleep(10) 

      last_transform = vehicle.get_transform()
      last_velocity = vehicle.get_velocity()
      # last_ang_vel = vehicle.get_angular_velocity()
      last_state = np.array([last_transform.location.x, 
                              last_transform.location.y, 
                              last_transform.rotation.yaw*np.pi/180, 
                              last_velocity.x, 
                              last_velocity.y])
      
      steer, throttle, done = mpc.mpc_run(curr_state=last_state)
      print("steer: ", steer[0])
      print("throttle: ", throttle[0])
      control.throttle = throttle[0]
      control.steer = steer[0]

      vehicle.apply_control(control)
  
      i = i + 1 
      
os.system("pkill -9 CarlaUE4") 