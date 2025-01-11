# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:02:57 2024

@author: timo_
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


class SmoothTrajectory:
    def __init__(self, waypoints, resolution):
        """
        Initialisiert die Trajektorien-Generator-Klasse.

        :param waypoints: Liste von Stützstellen [(x1, y1), (x2, y2), ...]
        :param resolution: Abstand zwischen den Punkten in der generierten Trajektorie
        """
        self.waypoints = np.array(waypoints)
        self.resolution = resolution
        self.x = None
        self.y = None
        self.yaw = None
        self.total_length = None  # Gesamtlänge der Trajektorie

    def generate(self):
        """
        Generiert eine glatte Trajektorie durch Spline-Interpolation zwischen den Stützstellen.
        """
        if len(self.waypoints) < 2:
            raise ValueError("Mindestens zwei Stützstellen erforderlich.")

        # Extrahiere X- und Y-Koordinaten der Stützstellen
        way_x, way_y = self.waypoints[:, 0], self.waypoints[:, 1]

        # Berechne die kumulierten Streckenlängen (Abstände zwischen Stützstellen)
        distances = np.sqrt(np.diff(way_x)**2 + np.diff(way_y)**2)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Kumulierte Länge

        # Erstelle einen Interpolationsbereich mit gleichmäßigen Abständen
        total_length = cumulative_distances[-1]
        interp_distances = np.arange(0, total_length, self.resolution)

        # Spline-Interpolation für X und Y über die kumulierte Länge
        spline_x = CubicSpline(cumulative_distances, way_x)
        spline_y = CubicSpline(cumulative_distances, way_y)

        # Generiere interpolierte Werte
        self.x = spline_x(interp_distances)
        self.y = spline_y(interp_distances)

        # Berechne Yaw-Winkel (Orientierung) aus den abgeleiteten Werten
        dx = spline_x.derivative()(interp_distances)
        dy = spline_y.derivative()(interp_distances)
        self.yaw = np.arctan2(dy, dx)

        # Berechne die Gesamtlänge der Trajektorie
        self.total_length = self.calculate_length()

    def calculate_length(self):
        """
        Berechnet die Gesamtlänge der generierten Trajektorie.

        :return: Gesamtlänge der Trajektorie
        """
        if self.x is None or self.y is None:
            raise ValueError("Die Trajektorie wurde noch nicht generiert.")

        # Berechne die Distanzen zwischen aufeinanderfolgenden Punkten
        distances = np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2)
        return np.sum(distances)

    def plot(self, pfeil_abstand=10, punkt_abstand=5):
        """
        Visualisiert die generierte Trajektorie und den Yaw-Winkel.
        """
        if self.x is None or self.y is None or self.yaw is None:
            raise ValueError("Die Trajektorie wurde noch nicht generiert. Rufen Sie `generate()` auf.")

        plt.figure(figsize=(8, 6))
        plt.plot(self.x[::punkt_abstand], self.y[::punkt_abstand], 'o', label="Trajektorie", alpha=0.5)

        for i in range(0, len(self.x), pfeil_abstand):
            dx = np.cos(self.yaw[i])
            dy = np.sin(self.yaw[i])
            plt.arrow(self.x[i], self.y[i], dx, dy, head_width=0.5, color="red", alpha=0.6)

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Glatte Trajektorie mit Orientierung (Yaw Angle)")
        plt.axis("equal")
        plt.legend()
        plt.grid()
        plt.show()

