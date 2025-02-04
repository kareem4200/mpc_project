# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:39:54 2024

@author: timo_
"""
import math
import numpy as np
import matplotlib.pyplot as plt

class Vehicle:
    def __init__(self, x, y, yaw, vel, delta):

        self.length = 2.338          #Vehicle length           (m)
        self.width = 1.381           #Vehicle width            (m)
        self.rear_to_wheel= 0.339    #Distance rear to axel    (m)
        self.wheel_length = 0.531    #Wheel length             (m)
        self.wheel_width = 0.125     #Wheel width              (m)
        self.track = 1.094           #Vehile track             (m)
        self.wheel_base = 1.686      #Wheel base               (m)
        self.x = x
        self.y = y
        self.r = 0
        self.delta = delta           #Steering angle           (radians)
        self.yaw = yaw               #Yaw angle                (radians)
        self.beta = 0
        self.beta_dot = 0
        self.vel = vel               #Velocity                 (m/s)
        self.lat_acc = 0             #Lateral acceleration     (m/s²)
        self.Caf = 2*20400           #Cornering stiffness front(N/rad)
        self.Car = 2*31900           #Cornering stiffness rear (N/rad)
        self.mass = 538+80           #Vehicle mass             (kg)
        self.lf = 0.9442             #Front axel to CG         (m)
        self.lr = 0.7417             #Rear axel to CG          (m)
        self.Iz = 430.166            #Moment of inertia        (kg.m2)
        
    def motion_model(self, dt):
        "Vehicle model"
        A = np.array([[-1 * (self.Caf + self.Car) / (self.mass * self.vel),
                       (- self.lf * self.Caf + self.lr * self.Car) / (self.mass * pow(self.vel, 2)) - 1],
                      [-1 * ((self.lf * self.Caf - self.lr * self.Car) / self.Iz),
                       -1 * ((pow(self.lf, 2) * self.Caf + pow(self.lr, 2) * self.Car) / (self.Iz * self.vel))]])
        B = np.array([[self.beta], [self.r]])
        C = np.array(
            [[(self.Caf * self.delta) / (self.mass * self.vel)], [((self.lf * self.Caf) / self.Iz) * self.delta]])

        AB = np.matmul(A, B)
        [[self.beta_dot], [r_dot]] = AB + C

        self.beta += self.beta_dot * dt
        self.r += r_dot * dt
        self.yaw += self.r * dt

        if self.yaw > np.pi:
            self.yaw -= 2 * np.pi
        elif self.yaw < -np.pi:
            self.yaw += 2 * np.pi

        x_dot = self.vel * np.cos(self.yaw + self.beta)
        y_dot = self.vel * np.sin(self.yaw + self.beta)
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.lat_acc = self.vel * (self.r + self.beta_dot)
        
#Start Value
x_0=0.7417
y_0=0
yaw_0=0
vel_0=0.1
delta_grad= 34                         #34 degree max (deegre)
delta=delta_grad*(math.pi/180)          #Steering angle in (radians)
dt=0.001                                #delta t

EgoFzg = Vehicle(x_0, y_0, yaw_0, vel_0, delta)

plotx = []
ploty = []
plotlataccel = []

iterration = 100000
for i in range(iterration):
    EgoFzg.motion_model(dt)
    plotx.append(EgoFzg.x-np.cos(EgoFzg.yaw)*0.7417)
    ploty.append(EgoFzg.y-np.sin(EgoFzg.yaw)*0.7417)
    plotlataccel.append(EgoFzg.lat_acc)
    print(EgoFzg.lat_acc)

#Radius of the perfect car/circel
radius = EgoFzg.wheel_base / math.tan(delta)


# Perfect circle
theta = np.linspace(0, 2 * np.pi, iterration)
circle_x = 0 + radius * np.cos(theta)
circle_y = y_0 + radius + radius * np.sin(theta)

# Plot
plt.plot(plotx, ploty, label="Vehicle trajectory")
plt.plot(circle_x, circle_y, label="Perfect Circle", linestyle="--")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Linear-on-track")
plt.legend()
#plt.gca().set_xlim(-5, 5)
#plt.gca().set_ylim(0, 7.5)
plt.axis("equal")
plt.show()
array=np.zeros(len(plotx))
for i in range(len(plotx)):
    array[i] = i
    i += 1
    
plt.plot(array, plotlataccel, label="a_y")
plt.xlabel("Time [ms]")
plt.ylabel("Accel. [m/s²]")
plt.title("Linear-on-track")
plt.legend()
plt.gca().set_xlim(0, iterration+100)
plt.gca().set_ylim(0, 15)
#plt.axis("equal")
plt.show()
    