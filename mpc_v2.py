import numpy as np
import math
from scipy.optimize import minimize

class MPC:
      def __init__(self, time_step, horizon, initial_state):
            self.time_step = time_step
            self.horizon = horizon
            self.current_wp_idx = 0
            self.wheel_base = 1.686
            self.state = initial_state
            self.steer_hist = np.zeros(horizon)
            self.throttle_hist = np.zeros(horizon)
            self.goal_reached = False
  
      def cost(self, u, x0, waypoints):
            cost = 0.0
            
            x, y, theta, v = x0
      
            prev_delta, prev_a = 0, 0
            
            for i in range(self.horizon):
                  delta, a = u[2 * i], u[2 * i + 1]
                  target_x, target_y = waypoints[self.current_wp_idx]
                  
                  x_next = x + v * np.cos(theta) * self.time_step
                  y_next = y + v * np.sin(theta) * self.time_step
                  theta_next = theta + (v / self.wheel_base) * math.tan(delta) * self.time_step
                  v_next = v + a * self.time_step
                  
                  cost += np.linalg.norm([x_next - target_x, y_next - target_y])
                  # cost += 0.2 * (delta ** 2 + a ** 2)
                  
                  if i > 0:
                        steering_diff = np.abs(delta - prev_delta)
                        throttle_diff = np.abs(a - prev_a)
                        cost += 0.2 * (steering_diff ** 2 + throttle_diff ** 2)
                  
                  prev_delta, prev_a = delta, a
                  
                  x, y, theta, v = x_next, y_next, theta_next, v_next
            print("Cost: ", cost)

            return cost
      
      def mpc_run(self, trajectory):
            
            bounds = [(-np.pi/4, np.pi/4), (-2.0, 2.0)] * self.horizon
            
            u0 = np.array([[self.steer_hist[i], self.throttle_hist[i]] for i in range(self.horizon)])
            
            res = minimize(
                  self.cost,
                  u0,
                  args=(self.state, trajectory.copy()),
                  method="SLSQP",
                  bounds=bounds,
            )

            u_opt = res.x
            u1 = u_opt[::2]  # Steering values (even indices)
            u2 = u_opt[1::2]  # Throttle values (odd indices)
                        
            self.steer_hist = u1
            self.throttle_hist = u2
            
            self.state[0] += self.state[3] * np.cos(self.state[2]) * self.time_step
            self.state[1] += self.state[3] * np.sin(self.state[2]) * self.time_step
            self.state[2] += (self.state[3] / self.wheel_base) * math.tan(u1[0]) * self.time_step
            self.state[3] += u2[0] * self.time_step
            
            if np.linalg.norm([self.state[0] - trajectory[self.current_wp_idx][0], 
                               self.state[1] - trajectory[self.current_wp_idx][1]]) <= 0.5:
                  self.current_wp_idx += 1
                  if self.current_wp_idx >= len(trajectory):
                        return self.state, u1, u2, True
            
            return self.state, u1, u2, False
