#!/usr/bin/env python
"""States Module 

Author: Hunter Mellema

Problem statement: 

Write a computer program “package” that will take an initial spacecraft position and
velocity vector, r_0, v_0, at an initial time t_0, and predict the future/past position and
velocity of the spacecraft at an arbitrary time t.

"""
import numpy as np
import math
from scipy.optimize import newton
from scipy.integrate import odeint

class Cartesian_State(object):
    """Stores parameters for cartesian state and methods for state conversion and propagation

    Args: 
    pos (numpy.array[float]): 3-vector of postitions
    vel (numpy.array[float]): 3-vector of velocities
    mu (float): gravitational parameter to use for propagation and conversion
    time (float): time at which state is defined
    
    """
    def __init__(self, pos, vel, mu, time):
        self.mu = mu 
        self.time = time # epoch 
        self.pos = pos #km 
        self.vel = vel #km/sec
        
    def __add__(self, other_state):
        """ Adds two Cartesian states together. Returns a state representing the result

        Returns:

        (Cartesian_State) = New state with only positions and velocities 
        """
        if not isinstance(other_state, 'Cartesian_State'):
            raise TypeError("Cartesian_State can only be added to another cartesian state!")
        
        pos_new = np.add(self.pos, other_state.pos)
        vel_new = np.add(self.vel, other_state.vel)
        
        return Cartesian_State(pos_new, vel_new)
        
    def to_kep_state(self):
        """ Converts Cartesian State into a Keplerian_State 

        Rtype: Keplerian_State
        """
        if not self.mu:
            raise ValueError("A gravitational parameter (MU) must exist for conversion")
        
        radius = np.linalg.norm(self.pos)
        velocity = np.linalg.norm(self.vel)
        v_radial = np.dot(self.pos, self.vel) / radius
        h_vec = np.cross(self.pos, self.vel)
        h = np.linalg.norm(h_vec)
        incl = math.acos(h_vec[2] / h)
        node_vec = np.cross(np.array([0, 0, 1]), h_vec)
        node_mag = np.linalg.norm(node_vec)
        if node_vec[1] >= 0:
            raan = math.acos(node_vec[0] / node_mag)
        else:
            raan = 2 * np.pi - math.acos(node_vec[0] / node_mag)
        ecc_vec = 1 / self.mu * ((velocity**2 - self.mu / radius) * self.pos -
                                 v_radial * radius * self.vel)
        ecc = np.linalg.norm(ecc_vec)
        if ecc_vec[2] >= 0:
            arg_peri = math.acos(node_vec / node_mag * ecc_vec / ecc)
        else:
            arg_peri = 2 * np.pi - math.acos(np.dot(node_vec / node_mag , ecc_vec / ecc))
        if v_radial >= 0:
            true_anomaly = math.acos(np.dot(ecc_vec / ecc, self.pos / radius))
        else:
            true_anomaly = 2 * np.pi * math.acos(ecc_vec / ecc * self.pos / radius)
            
        return Keplerian_State(radius, h, incl, raan, ecc, arg_peri,
                               true_anomaly, self.mu, self.time)
    
    def propagate(self, times):
        delta_ts = times - self.time

        # initial conditions
        q0 = np.concatenate((self.pos, self.vel))
        mu = self.mu
        def diff_eq(q, t):
            dstate_dt = [q[3], q[4], q[5],
                         -mu * q[0] / np.linalg.norm(q[0:3]),
                         -mu * q[1] / np.linalg.norm(q[0:3]),
                         -mu * q[2] / np.linalg.norm(q[0:3])]
            return dstate_dt
            
        m = odeint(diff_eq, q0, delta_ts)

        print(m)
        
        #pos_new = np.array([x_new, y_new, z_new])
        #vel_new = np.array([vx_new, vy_new, vz_new])

        #return Cartesian_State(pos_new, vel_new, self.mu, t_next)
        
    
    
class Keplerian_State(object):
    """Stores parameters for keplerian state and methods for state conversion and propagation

    Args: 

    

    """
    def __init__(self, radius, h, incl, raan, ecc, arg_peri, true_anom, mu, time):
        self.radius = radius
        self.h = h
        self.incl = incl
        self.raan = raan
        self.ecc = ecc
        self.arg_peri = arg_peri
        self.true_anom = true_anom
        self.mu = mu
        self.time = time
        self.a = self.h**2 / (self.mu * (1 - self.ecc))
        self.n = math.sqrt(self.mu / self.a**3)
        
    @property
    def eccentric_anomaly(self):
        try:
            return self._ecc_anom
        except AttributeError:
            self._ecc_anom = 2 * math.atan2(math.sqrt(1 - self.ecc) *
                                            math.tan(self.true_anom / 2),
                                            math.sqrt(1 + self.ecc))
        return self._ecc_anom
    
    @property
    def mean_anomaly(self):
        try:
            return self._mean_anom
        except AttributeError:
            self._mean_anom = self.eccentric_anomaly - self.ecc * math.sin(self.eccentric_anomaly)
            return self._mean_anom
        
    @property
    def ts_peri(self):
        """ Time since perigee """ 
        try:
            return self._ts_peri
        except AttributeError:
            self._ts_peri = self.mean_anomaly * self.radius**(3 / 2) / math.sqrt(self.a**3 / self.mu)
            return self._ts_peri
        
    def to_cart_state(self):
        """ """
        pass 
    
    def propagate(self, new_time):
        """ """ 
        delta_t = new_time - self.time
        new_mean_anom = self.n * (new_time - self.ts_peri)

        def kep_eq_solver(E): 
            return  - E + self.ecc * math.sin(E)

        # solves using newton's method with mean anomaly as initial guess
        E_new = newton(kep_eq_solver, new_mean_anom)

        ta_new = math.acos((math.cos(E_new) - self.ecc) / (1 - self.ecc * math.cos(E_new)))

        r_new = self.h**2 / self.mu * 1 / ( 1 + self.ecc * math.cos(ta_new))
        
        return Keplerian_State(r_new, self.h, self.incl, self.raan, self.ecc,
                               self.arg_peri, ta_new, self.mu, new_time)
