#!/usr/bin/env python
"""States Module 

Author: Hunter Mellema
Summary: Provides Cartesian_State and Keplerian_State objects for representing and
propagating orbits in 2 body systems

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
        self.pos = pos #km 
        self.vel = vel #km/sec
        self.mu = mu 
        self.time = time # epoch 

    @property
    def h(self):
        """ Specific angular momentum of the orbit 

        Returns: 
        h (float): specific angular momentum of the orbit (km^2/seconds)
        """ 
        try:
            return self._h
        except AttributeError:
            self._h = np.linalg.norm(np.cross(self.pos, self.vel))
            return self._h

    @property
    def nrg(self):
        """ Specific energy of the orbit 

        Returns: 
        nrg (float): specific energy of the orbit (km^2/sec^2)
        """ 
        try:
            return self._nrg
        except AttributeError:
            self._nrg = np.linalg.norm(self.vel)**2 / 2 - self.mu / np.linalg.norm(self.pos)
            return self._nrg
        
    def to_kep_state(self):
        """ Converts Cartesian State into a Keplerian_State 

        Returns: 
        (Keplerian_State): kep state for the same orbit assuming 2 body dynamics
        """
        if not self.mu:
            raise ValueError("A gravitational parameter (MU) must exist for conversion")
        
        radius = np.linalg.norm(self.pos)
        velocity = np.linalg.norm(self.vel)
        v_radial = np.dot(self.pos, self.vel) / radius

        # find angular momentum 
        h_vec = np.cross(self.pos, self.vel)
        h = np.linalg.norm(h_vec)

        # find inclination
        incl = math.acos(h_vec[2] / h)

        # calculate node 
        node_vec = np.cross(np.array([0, 0, 1]), h_vec)
        node_mag = np.linalg.norm(node_vec)

        # calculate raan 
        if node_mag == 0:
            # this is the circular orbit case
            raan = np.nan
        elif node_vec[1] >= 0:
            raan = math.acos(node_vec[0] / node_mag)
        else:
            raan = 2 * np.pi - math.acos(node_vec[0] / node_mag)

        # calculate eccentricity
        ecc_vec = 1 / self.mu * ((velocity**2 - self.mu / radius) * self.pos -
                                 v_radial * radius * self.vel)
        ecc = np.linalg.norm(ecc_vec)

        # calculate argument of perigee 
        if ecc == 0 or node_mag == 0:
            #circular orbit case
            arg_peri = np.nan
        elif ecc_vec[2] >= 0:
            arg_peri = math.acos(np.dot(node_vec / node_mag, ecc_vec / ecc))
        else:
            arg_peri = 2 * np.pi - math.acos(np.dot(node_vec / node_mag , ecc_vec / ecc))

        # calculate true anomaly 
        if ecc == 0 or node_mag == 0:
            true_anomaly = np.nan
        elif v_radial >= 0:
            true_anomaly = math.acos(np.dot(ecc_vec / ecc, self.pos / radius))
        else:
            true_anomaly = 2 * np.pi * math.acos(np.dot(ecc_vec / ecc, self.pos / radius))
            
        return Keplerian_State(radius, h, incl, raan, ecc, arg_peri,
                               true_anomaly, self.mu, self.time)
    
    def propagate(self, time_new, spacing=100):
        """ Propagates state forward (or backwards) in time to new time
        
        Args: 
        time_new (float): new time to propagate to 

        Returns: 
        (Cartesian_State): new cartesian state at new time 
        """
        delta_ts = np.linspace(0, time_new - self.time, spacing)

        # initial conditions = state coordinates
        q0 = np.concatenate((self.pos, self.vel))

        # use diff equation solver in python to numerically integrate
        def diff_eq(q, t):
            x, y, z, dx, dy, dz = q
            r = math.sqrt(x**2 + y**2 + z**2)
            dstate_dt = [dx, dy, dz,
                         -self.mu * x / r**3,
                         -self.mu * y / r**3,
                         -self.mu * z / r**3]
            return dstate_dt    
        new_states = odeint(diff_eq, q0, delta_ts)

        return Cartesian_State(new_states[-1][0:3], new_states[-1][3:], self.mu, time_new)

    
class Keplerian_State(object):
    """Stores parameters for keplerian state and methods for state conversion and propagation

    Args: 
    radius (float): radius of the orbit at the specified time (km)
    h (float): specific angular momentum of the orbit (km)
    incl (float): inclination of the orbit in (radians)
    raan (float): right ascension of the ascending node(radians)
    ecc (float): orbit eccentric (non-dimensional, defined on ecc>=0)
    arg_peri (float): argument of perigee (radians)
    true_anom (float): true anomaly (radians)
    mu (float): gravitational parameter of central body (km^3/s^2)
    time (float): time at which state is defined (sec)
    
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

        # internally calculated parameters 
        self.a = self.h**2 / (self.mu * (1 - self.ecc**2))
        self.n = math.sqrt(self.mu / self.a**3)
        
    @property
    def eccentric_anomaly(self):
        """ Eccentric anomaly of the orbit calculated from true anomaly

        Returns: 
        eccentric_anomaly (float): eccentric anomaly of orbit in radians
        """
        try:
            return self._ecc_anom
        except AttributeError:
            self._ecc_anom = 2 * math.atan2(math.sqrt(1 - self.ecc) *
                                            math.tan(self.true_anom / 2),
                                            math.sqrt(1 + self.ecc))
        return self._ecc_anom
    
    @property
    def mean_anomaly(self):
        """ Mean anomaly of the orbit. Calculated lazily 

        Returns: 
        mean_anomaly (float): mean anomaly in radians
        """ 
        try:
            return self._mean_anom
        except AttributeError:
            self._mean_anom = self.eccentric_anomaly - self.ecc * math.sin(self.eccentric_anomaly)
            return self._mean_anom
        
    @property
    def ts_peri(self):
        """ Time since perigee 

        Returns: 
        ts_peri (float): time of periapsis passage in seconds
        """ 
        try:
            return self._ts_peri
        except AttributeError:
            self._ts_peri = self.time - math.sqrt(self.a**3 / self.mu ) * (self.eccentric_anomaly -
                                                                           self.ecc *
                                                                           math.sin(self.eccentric_anomaly))
            #self.mean_anomaly * self.radius**(3 / 2) / math.sqrt(self.a**3 / self.mu)
            return self._ts_peri

    @property
    def nrg(self):
        """ Returns the specific energy of this orbit 

        Returns: 
        nrg (float): specific energy of the orbit in km^2/sec^2
        """
        try:
            return self._nrg
        except AttributeError:
            self._nrg = - self.mu / (2 * self.a)
            return self._nrg
    
    @property
    def cart_pos(self):
        """ Gives the cartesian position vector of the Keplerian_State in inertial coords
        
        Returns: 
        (numpy.array(float)): coordinate vector of state position (km)
        """
        try:
            return self._cart_pos
        except AttributeError:
            self._cart_pos = np.asarray(np.matmul(self.q_matrix(), self.perifocal_pos()))[0]
            return self._cart_pos

    @property
    def cart_vel(self):
        """ Gives the cartesian velocity vector of the Keplerian_State in inertial coords
        
        Returns: 
        (numpy.array(float)): coordinate vector of state velocity (km/sec)
        """
        try:
            return self._cart_vel
        except AttributeError:
            self._cart_vel = np.asarray(np.matmul(self.q_matrix(), self.perifocal_vel()))[0]
            return self._cart_vel 

    @property
    def period(self):
        """ Computes the orbital period of the satellite (assumes elliptical orbit) 

        Returns: 
        (float): orbit period (seconds)
        """
        return 2 * np.pi / math.sqrt(self.mu) * math.sqrt(self.a**3)

        
    def q_matrix(self):
        """ Calculates q matrix for converting from perifocal coordinates to inertial coordinates

        returns: 
        (numpy.Matrix)
        """
        q = np.matrix([[-math.sin(self.raan) * math.cos(self.incl) * math.sin(self.arg_peri) + math.cos(self.raan) * \
                        math.cos(self.arg_peri),
                        -math.sin(self.raan) * math.cos(self.incl) * math.cos(self.arg_peri) -  math.cos(self.raan) * \
                        math.sin(self.arg_peri),
                        math.sin(self.raan) * math.sin(self.incl)],
                       [math.cos(self.raan) * math.cos(self.incl) * math.sin(self.arg_peri) + math.sin(self.raan) * \
                        math.cos(self.arg_peri),
                        math.cos(self.raan) * math.cos(self.incl) * math.cos(self.arg_peri) - math.sin(self.raan) * \
                        math.sin(self.arg_peri),
                        -math.cos(self.raan) * math.sin(self.incl)],
                       [math.sin(self.incl) * math.sin(self.arg_peri),
                        math.sin(self.incl) * math.cos(self.arg_peri),
                        math.cos(self.incl)]])
        return q 
    
    def perifocal_pos(self):
        """ Finds position in perifocal coordinate frame
        
        Returns: 
        (numpy.array(float)): position 3-vector in perifocal frame
        """
        pos = self.radius * np.array([math.cos(self.true_anom), math.sin(self.true_anom), 0])
        return pos

    def perifocal_vel(self):
        """ Finds velocity in perifocal coordinate frame
        
        Returns: 
        (numpy.array(float)): velocity 3-vector in perifocal frame
        """
        vel = self.mu / self.h * np.array([-math.sin(self.true_anom), self.ecc + math.cos(self.true_anom), 0])
        return vel

    def prop_circular(self, new_time):
        """ Propagates the state up to the specified time

        Args: 
        new_time (float): time to propagate up to in seconds

        Returns: 
        (Keplerian_State): new keplerian state at time=new_time

        Notes:
        - Does NOT work with non-zero eccentricity orbits
        """
        if ecc != 0:
            raise ValueError("This method ONLY works for ECC = 0!")
        mean_anomaly = self.n * (new_time - self.ts_peri)

        new_state = self.copy()
        new_state.true_anom = mean_anomaly
        new_state.time = new_time
        return new_state
        
    def propagate(self, new_time):
        """ Propagates the state up to the specified time

        Args: 
        new_time (float): time to propagate up to in seconds

        Returns: 
        (Keplerian_State): new keplerian state at time=new_time

        Notes:
        - Does NOT work with zero eccentricity orbits
        """
        if self.ecc == 0 or np.isnan(self.raan):
            raise ValueError("Cannot propagate zero-eccentricity/circular orbits. Call: \n" +
                             "Keplerian_State.prop_circular(*args) instead")
        
        new_mean_anom = self.n * (new_time - self.ts_peri)

        # solves using newton's method with mean anomaly as initial guess
        def kep_eq_solver(E): 
            return  E + self.ecc * math.sin(E) - new_mean_anom
        
        E_new = newton(kep_eq_solver, new_mean_anom)

        # re-calculate true anomaly and radius at new time
        ta_new = 2 * math.atan2(math.tan(E_new / 2) , math.sqrt((1 - self.ecc) / (1 + self.ecc)))

        r_new = self.h**2 / self.mu * 1 / ( 1 + self.ecc * math.cos(ta_new))
        
        return Keplerian_State(r_new, self.h, self.incl, self.raan, self.ecc,
                               self.arg_peri, ta_new, self.mu, new_time)
