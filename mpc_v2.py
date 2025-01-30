import numpy as np
import math
from scipy.optimize import minimize

class MPC:
      def __init__(self, time_step, horizon, trajectory):
            self.time_step = time_step
            self.horizon = horizon
            self.current_wp_idx = 0
            self.wheel_base = 2.875 # tesla
            self.yaw_rate = 0
            self.max_steer = 45
            self.max_acc = 2
            # self.state = initial_state
            self.steer_hist = np.zeros(horizon)
            # self.throttle_hist = np.zeros(horizon)
            # self.throttle_hist = np.full(shape=horizon, fill_value=0.5)
            self.trajectory = trajectory
            self.goal_reached = False
  
      def cost(self, u, waypoints, currunt_xy):
            cost = 0.0
            
            x, y, theta, vx, vy = currunt_xy
      
            prev_delta, prev_a = 0, 0

            for i in range(self.horizon):
                  target_x, target_y = waypoints[self.current_wp_idx]
                  
                  delta = (u[i] * self.max_steer) * math.pi / 180
                  
                  x_next = x + vx * self.time_step
                  y_next = y + vy * self.time_step
                  theta_next = theta + (math.sqrt(vx**2 + vy**2) / self.wheel_base) * math.tan(delta) * self.time_step
                  v_next = math.sqrt(vx**2 + vy**2) + self.max_acc * self.time_step
             
                  cost += np.linalg.norm([x_next - target_x, y_next - target_y])
                  cost += 0.2 * (delta ** 2)
                  
                  if i > 0:
                        steering_diff = np.abs(delta - prev_delta)
                        cost += 0.6 * (steering_diff ** 2)
                  
                  prev_delta = delta
                  
                  x, y, theta, vx, vy = x_next, y_next, theta_next, v_next*math.cos(theta_next), v_next*math.sin(theta_next)

            return cost
      
      def mpc_run(self, curr_state):
            
            # bounds = [(-0.1, 0.1), (0.0, 1.0)] * self.horizon
            bounds = [(-1.0, 1.0)] * self.horizon
            
            u0 = np.array([[self.steer_hist[i]] for i in range(self.horizon)])
            
            res = minimize(
                  self.cost,
                  u0,
                  args=(self.trajectory.copy(), curr_state),
                  method="SLSQP",
                  bounds=bounds,
            )

            u_opt = res.x
       
            if np.linalg.norm([curr_state[0] - self.trajectory[self.current_wp_idx][0], 
                               curr_state[1] - self.trajectory[self.current_wp_idx][1]]) <= 1.0:
                  self.current_wp_idx += 1
                  if self.current_wp_idx >= len(self.trajectory):
                        return u_opt, True
            
            return u_opt, False
