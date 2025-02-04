# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:15:02 2025

@author: timo_
"""
import math
import numpy as np
import matplotlib.pyplot as plt

class VehicleModel:
    def __init__(self, x, y, yaw, vel):
        
        """ Fahrzeugparameter """
        self.m = 538+80             # Masse des Fahrzeugs [kg]
        self.Iz = 430.166           # Trägheitsmoment um die z-Achse [kg·m^2]
        self.lv = 0.9442            # Abstand Schwerpunkt - Vorderachse [m]
        self.lh = 0.7417            # Abstand Schwerpunkt - Hinterachse [m]
        self.l = 1.686              # Wheel base [m]
        self.hS = 0.54              # Höhe Schwerpunkt [m]
        self.g = 9.81
        self.Caf = 20400          # Cornering stiffness front(N/rad)
        self.Car = 31900          # Cornering stiffness rear (N/rad)

        self.rv = 1.7
        self.rh = 1.7
        
        #self.Fzv = (self.lh / self.l) * self.m * self.g
        
        """ Reifen """
        self.muLv_max = 1.3
        self.muQv_max = 1.3
        self.mu_Lh_max = 1.3
        self.mu_Qh_max = 1.3
        self.bLv = 11.4
        self.cLv = 1.5
        self.bQv = 7.5
        self.cQv = 1.3
        self.b_Lh = 16.3
        self.c_Lh = 1.5
        self.b_Qh = 15.3
        self.c_Qh = 1.3
        self.r = 0.25               # Radius Rad [m]
        self.dLh = 0.011            # Reifenlatsch [m]
        self.dQh = 0.011            # Reifenlatsch [m]
        self.dLv = 0.011            # Reifenlatsch [m]
        self.dQv = 0.011            # Reifenlatsch [m]
        
        """ Initial State """
        self.v = vel
        self.omega_v = self.v / self.r
        self.omega_h =self.v/self.r
        self.psi = yaw
        self.psi_dot = 0
        self.beta = 0
        self.rX = x
        self.rY = y
        self.ax = 0
        self.ax_prev = 0
        self.ay = 0
        self.FLv_dyn = 0
        self.FQv_dyn = 0
        self.FLh_dyn = 0
        self.FQh_dyn = 0
        self.FLh = 0
        self.FLv = 0
        

        
        
        self.Zeitschritt = 0                #Debug
        
    def updateState(self, dt, delta_v, delta_h=0):
        self.Zeitschritt += 1
        """ Berechnung Laengsschlupf """
        omega_v = self.omega_v+dt*(1/self.lv)*(-self.FLv*self.r)
        omega_h = self.omega_h+dt*(1/self.lh)*(-self.FLh*self.r)
        #omega_v_next = omega_v + T / Iv * (MAv - MBv - FLv * self.r)
        #omega_h_next = omega_h + T / Ih * (self.MAh - self.MBh - FLh * self.r)
        vLv = self.v * math.cos(self.beta) * math.cos(delta_v) + self.v * math.sin(self.beta) + self.psi_dot * self.lv * math.sin(delta_v)
        vLh = self.v * math.cos(self.beta) * math.cos(delta_h) + self.v * math.sin(self.beta) - self.psi_dot * self.lh * math.sin(delta_h)
    
        if (max(abs(omega_v * self.r), abs(vLv)) == 0):
            sLv = 0
        else:
            sLv = (omega_v * self.r - vLv) / max(abs(omega_v * self.r), abs(vLv))
        if (max(abs(omega_h * self.r), abs(vLh)) == 0):
            sLh = 0 
        else:
            sLh = (omega_h * self.r - vLh) / max(abs(omega_h * self.r), abs(vLh))

        
        """ Berechnung Querschluspfwerte """
        # Vorderachse
        alpha_v = delta_v - math.atan2(self.v * math.sin(self.beta) + self.psi_dot * self.lv, self.v * math.cos(self.beta))
        #sQv = math.tan(alpha_v)
        sQv = np.sign(alpha_v)*min(1, abs(math.tan(alpha_v)))
        #Hinterachse
        alpha_h = delta_h - math.atan2(self.v * math.sin(self.beta) - self.psi_dot * self.lh, self.v * math.cos(self.beta))
        #sQh = math.tan(alpha_h)
        sQh = np.sign(alpha_h)*min(1, abs(math.tan(alpha_h)))

        if (abs(self.v) < 10**-3):
            alpha_v = 0
            alpha_h = 0
            sQv = 0
            sQh = 0
            
        """ Berechnung Laengs- und Querkraefte """
        # Normalkraft an der Vorderachse
        Fzv = (self.lh / self.l) * self.m * self.g - (self.hS/self.l) * self.m * self.ax_prev
        
        sv_star = np.sqrt(sLv**2 + sQv**2)# + epsilon
        FLv_star = Fzv * self.muLv_max * math.sin(self.cLv * math.atan2(self.bLv * sv_star , self.muLv_max))
        FQv_star = Fzv * self.muQv_max * math.sin(self.cQv * math.atan2(self.bQv * sv_star , self.muQv_max))
        print("sv_star:", sv_star, "Damit FQv_star (nicht linear):", FQv_star, "Mit lienarer Annahme", self.Caf*sv_star )
        
        if (sv_star < 10**-12):
            self.FLv = 0
            FQv = 0
        else:
            Fv_star = math.sqrt(((sLv/sv_star) * FLv_star)**2 + ((sQv/sv_star) * FQv_star)**2)
            self.FLv = (sLv / sv_star) * Fv_star
            FQv = (sQv / sv_star) * Fv_star
        
        FLv_dyn_prev = self.FLv_dyn
        self.FLv_dyn = self.FLv + math.exp(-dt * abs(omega_v) * self.r / self.dLv) * (FLv_dyn_prev - self.FLv)
        FQv_dyn_prev = self.FQv_dyn
        self.FQv_dyn = FQv + math.exp(-dt * abs(omega_v) * self.r / self.dQv)* (FQv_dyn_prev - FQv)
        

        """ Kraefte Hinterachse """
        Fzh = (self.lv / self.l) * self.m * self.g + (self.hS/self.l) * self.m * self.ax_prev
        
        sh_star = np.sqrt(sLh**2 + sQh**2) #+ epsilon
        FLh_star = Fzh * self.mu_Lh_max * np.sin(self.c_Lh * np.arctan2(self.b_Lh * sh_star , self.mu_Lh_max))
        FQh_star = Fzh * self.mu_Qh_max * np.sin(self.c_Qh * np.arctan2(self.b_Qh * sh_star , self.mu_Qh_max))
        
        if(sh_star< 10**-12):
            self.FLh = 0
            FQh = 0
        else:
            Fh_star = np.sqrt(((sLh/sh_star) * FLh_star)**2 + ((sQh/sh_star) * FQh_star)**2)
            self.FLh = (sLh / sh_star) * Fh_star 
            FQh = (sQh / sh_star) * Fh_star

        FLh_dyn_prev = self.FLh_dyn
        self.FLh_dyn = self.FLh + np.exp(-dt * abs(omega_h) * self.r / self.dLh) * (FLh_dyn_prev - self.FLh)
        FQh_dyn_prev = self.FQh_dyn
        self.FQh_dyn = FQh + np.exp(-dt * abs(omega_h) * self.r / self.dQh) * (FQh_dyn_prev - FQh)

        #Gesamt
        self.Fx = (
            self.FLv_dyn * np.cos(delta_v) - self.FQv_dyn * np.sin(delta_v) +
            self.FLh_dyn * np.cos(delta_h) - self.FQh_dyn * np.sin(delta_h))
        self.Fy = (
            self.FLv_dyn * np.sin(delta_v) + self.FQv_dyn * np.cos(delta_v) +
            self.FLh_dyn * np.sin(delta_h) + self.FQh_dyn * np.cos(delta_h))
        
        

        """ Berechnung Zustandsgroessen """
        v_dot = (1 / self.m) * (self.Fx * np.cos(self.beta) + self.Fy * np.sin(self.beta))
        self.v += v_dot*dt
        
        psi_dot_dot = (1 / self.Iz) * (
            (self.lv * ((self.FLv_dyn * np.sin(delta_v)) + (self.FQv_dyn * np.cos(delta_v)))) -
            (self.lh * ((self.FLh_dyn * np.sin(delta_h)) + (self.FQh_dyn * np.cos(delta_h))))
        )
        
        beta_dot = (1 / (self.m * self.v)) * ((self.Fy * np.cos(self.beta)) - self.Fx * np.sin(self.beta)) - self.psi_dot


        # Aktualisierung der Zustände
        self.beta += beta_dot * dt
        self.psi_dot += psi_dot_dot * dt
        self.psi += self.psi_dot * dt + 0.5 * psi_dot_dot * dt**2
        
        # if self.psi > np.pi:
        #     self.psi -= 2 * np.pi
        # elif self.psi < -np.pi:
        #     self.psi += 2 * np.pi

 
        #self.ay = self.v * beta_dot + self.psi_dot * self.v
        self.ay=self.Fy/self.m
        self.ax_prev=self.ax
        self.ax=self.Fx/self.m
        self.rX += self.v * np.cos(self.psi + self.beta) * dt +0.5 * self.ax * np.cos(self.psi) *dt**2 - 0.5 * self.ay * np.sin(self.psi) * dt**2
        self.rY += self.v * np.sin(self.psi + self.beta) * dt +0.5 * self.ax * np.sin(self.psi) *dt**2 + 0.5 * self.ay * np.cos(self.psi) * dt**2
        
        if (self.Zeitschritt%2 == 0):
            print("----- Zeitschritt:", self.Zeitschritt, " -----")
            print("Fx:", self.Fx, "Fy:", self.Fy)
            print("Yaw [degree]:", np.rad2deg(self.psi), "Yaw-rate [degree]/s:", np.rad2deg(self.psi_dot))
            print("Geschw. [m/s]:", self.v)
            print("Querbesch. [m/s²]:", EgoFzg.ay)
        
#Start Value
x_0=0
y_0=0
yaw_0=0
vel_0= 1
delta=np.deg2rad(34)
delta_h=0
dt=0.001                                #delta t

EgoFzg = VehicleModel(x_0, y_0, yaw_0, vel_0)

plotx = []
ploty = []
plotrx = []
plotry = []
plotlataccel = []

iterration = 12000
for i in range(iterration):
    EgoFzg.updateState(dt, delta, delta_h=0)
    #plotx.append(EgoFzg.x)
    #ploty.append(EgoFzg.y)
    plotrx.append(EgoFzg.rX-np.cos(EgoFzg.psi)*0.7417)
    plotry.append(EgoFzg.rY-np.sin(EgoFzg.psi)*0.7417)
    plotlataccel.append(EgoFzg.ay)
    #print("Querbesch. [m/s²]:", EgoFzg.ay)

#Radius of the perfect car/circel
radius = EgoFzg.l / math.tan(delta)


# # Perfect circle
theta = np.linspace(0, 2 * np.pi, iterration)
circle_x = -0.7417 + radius * np.cos(theta)
circle_y = y_0 + radius + radius * np.sin(theta)

# Plot
#plt.plot(plotx, ploty, label="Vehicle trajectory")
plt.plot(plotrx, plotry, label="Vehicle trajectory")
plt.plot(circle_x, circle_y, label="Perfect Circle", linestyle="--")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Non-linear-one-track")
plt.legend()
#plt.gca().set_xlim(-5, 5)
#plt.gca().set_ylim(0, 7.5)
plt.axis("equal")
plt.show()

array=np.zeros(len(plotrx))
for i in range(len(plotrx)):
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

