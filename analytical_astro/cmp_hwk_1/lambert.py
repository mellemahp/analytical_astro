#!/usr/bin/env python
"""Lambert Module

Author: Hunter Mellema
Summary: Provides a lambert_solver object that takes in 2 Keplerian_State objects and
a transfer time and calculates a family of possible transfer solutions

Note: the first algorithm used here is taken from "Lambert's Problem" by Kate Davis from CCAR

"""
import numpy as np
import math
from scipy.optimize import newton

class LamberSolverCartesian(object):
    """ """ 
    def __init__(self, initial_state, final_state):
       # if not isinstance(initial_state, Cartesian_State) or not isinstance(final_state, Cartesian_State):
        #    raise TypeError("The Lambert Cartesian Solver can only accept Cartesian_State objects")
        if not initial_state.mu == final_state.mu:
            raise ValueError("Both states MUST have the same gravitaional parameter: mu")

        self.i_state = initial_state
        self.f_state = final_state

    def solve(self, delta_t_0, f_in_orb_seg=False):
        """Solves laberts problem using only positions and a desire
        
        Args: 
        delta_t_0 (float): 
        f_in_orb_seg (bool): if false returns orbit segment that does not contain empty focus
        
        """
        c = np.linalg.norm(self.i_state.pos - self.f_state.pos)
        r_1 = np.linalg.norm(self.i_state.pos)
        r_2 = np.linalg.norm(self.f_state.pos)
        s = 1 / 2 * (r_1 + r_2 + c)

        a_min = s / 2


        # check for hyperbolic orbit
        if f_in_orb_seg: 
            delta_t_p = 1 / (6 * math.sqrt(self.i_state.mu)) * ((2*s)**(3/2) + (r_1 + r_2 - c)**(3/2))
        else: 
            delta_t_p = 1 / (6 * math.sqrt(self.i_state.mu)) * ((2*s)**(3/2) - (r_1 + r_2 - c)**(3/2))
        
        if delta_t_0 < delta_t_p:
            category = 'Elliptic'
        else:
            category = 'Hyperbolic'

        def alpha(a):
            if f_in_orb_seg: 
                return 2 * math.asin(math.sqrt(s / (2 * a)))
            else:
                return 2 * np.pi - 2 * math.asin(math.sqrt(s / (2 * a)))
            
        def beta(a):
            if f_in_orb_seg: 
                return 2 * math.asin(math.sqrt((s - c) / (2 * a)))
            else:
                return -2 * math.asin(math.sqrt((s - c) / (2 * a)))
        
        def lamberts_problem(a):
            return a**(3 / 2) * (alpha(a) - beta(a) - (math.sin(alpha(a)) - math.sin(beta(a)))) - \
                math.sqrt(self.i_state.mu) * delta_t_0

        a = newton(lamberts_problem, a_min)
            
        # computing velocities:
        A = math.sqrt(self.i_state.mu / (4 * a)) / math.tan(alpha(a) / 2)
        B = math.sqrt(self.i_state.mu / (4 * a)) / math.tan(beta(a) / 2)
            
        c_hat = (self.f_state.pos - self.i_state.pos) / np.linalg.norm(self.f_state.pos - self.i_state.pos)
        r_1hat = self.i_state.pos / np.linalg.norm(self.i_state.pos)
        r_2hat = self.f_state.pos / np.linalg.norm(self.f_state.pos)
        
        v_1 = (B + A) * c_hat + (B - A) * r_1hat
        v_2 = (B + A) * c_hat - (B - A) * r_2hat
            
        return {'category':category, 'v_1':v_1, 'v_2':v_2}
        
    def min_energy(self):
        """Gives the minimum energy orbit (hohmann)

        Returns: 
        (float)
        """
        c = np.linalg.norm(self.i_state.pos - self.f_state.pos)
        r_1 = np.linalg.norm(self.i_state.pos)
        r_2 = np.linalg.norm(self.f_state.pos)
        s = 1 / 2 * (r_1 + r_2 + c)

        # at the minimum eccentrity major axis is parallel to chord, so:P
        a_em = (r_1 + r_2) / 2

        alpha = 2 * math.asin(math.sqrt(s / (2 * a_em)))
        beta = 2 * math.asin(math.sqrt((s - c) / (2 * a_em)))
        t_1 = 1 / (math.sqrt(self.i_state.mu)) * a_em**(3 / 2) * (alpha - beta - (math.sin(alpha) -
                                                                                  math.sin(beta)))
        
        # modify alpha and beta to go the other way
        beta *= -1
        alpha = 2 * np.pi - alpha
        t_2 = 1 / (math.sqrt(self.i_state.mu)) * a_em**(3 / 2) * (alpha - beta - (math.sin(alpha) -
                                                                                  math.sin(beta)))

        return {'a_em': a_em, 't_1': t_1 , 't_2': t_2} 

    def min_eccentricity(self):
        """ 
        """

        return
        

class LambertSolverKeplerian(object):
    """ Initializes a labert solver with a starting state and ending state to use for computing a transfer

    Args:
    initial_state (Keplerian_State): initial state to use as starting point for transfer
    final_state (Keplerian_State): final state to use as end point of transfer

    Raises: 
    TypeError: must input two Keplerian_State objects 
    ValueError ("Both States must... ): both input states must have the same gravitaional parameter
    ValueError ("Both A variable and.."): the states cannot have the same true anomaly and must have non-zero radii 

    Notes:
    - initial_state and final_state MUST have the same mu (gravitaional parameters)
    - Does not RUN solver until solve() method is called
    """
    def __init__(self, initial_state, final_state):
        if not isinstance(initial_state, Keplerian_State) or not isinstance(final_state, Keplerian_State):
            raise TypeError("The Lambert Solver can only accept Keplerian_State objects")
        if not initial_state.mu == final_state.mu:
            raise ValueError("Both states MUST have the same gravitaional parameter: mu")

        self.i_state = initial_state
        self.f_state = final_state

        #pre-initalize values that dont change if solve is called with different parameters
        self.delta_true_anom = self.final_state.true_anom - self.i_state.true_anom
        self.A = math.sqrt(self.i_state.radius * self.f_state.radius * (1 + math.cos(self.delta_true_anom)))

        if self.A == 0 or self.delta_ecc_anom == 0:
            raise ValueError("Both A variable and delta_true_anom must NOT = 0")

    def solve(self, delta_t_0, direction_of_motion='Auto', thresh=1 * 10**-6):
        """Solves a single trajectory between the start and end states

        Args:
        delta_t_0 (float): desired tranfer time (sec)
        direction_of_motion (dict)[optional]: select what direction of motion you would like to solve for (default='Auto')
        thresh (float)[optional]: sets threshold for convergence to desired transfer time (default=1 * 10**-6)

        Returns: 
        tuple(float, float, float): tuple of delta_v at initial state and final state and time to complete transfer trajectory

        Note:
        - For the direction of motion "Auto" will select the direction of motion based on what will give shortest transfer
        if you set "Neg" it will always travel in the negative direction and "Pos" will always travel in the positive direction
        """
        DM_OPTIONS = {'Neg':-1, 'Pos':1, 'Auto':self.__auto_dm()}

        # change sign of A based on selected direction of motion
        A = self.A * DM_OPTIONS[direction_of_motion]

        # set initial guesses (also pre-initalizes loop)
        c2 = 1 / 2
        c3 = 1 / 6
        psi = 0
        psi_up = 4 * np.pi
        psi_low = -4 * np.pi

        # set delta_t high enough to enter loop we will recompute it in the loop
        delta_t = delta_t_0 + 100

        while math.fabs(delta_t - delta_t_0) > thresh:
            y = self.__get_y(A, psi, c_2, c_3)
            
            # Change psi_low in order to prevent errors in sqrt later
            if A > 0 and y < 0:
                psi_low = self.__adjust_psi(A, psi, c_2, c_3)
                y = self.__get_y(A, psi_low, c_2, c_3)

            chi = math.sqrt(y / c_2)
            delta_t = self.__get_delta_t(chi, c_3, A, y)

            # bisect psi
            if delta_t <= delta_t_0:
                psi_low = psi
            else:
                psi_up = psi

            psi = (psi_up + psi_low) / 2

            # re-compute c_2, c_3
            if psi > thresh:
                c_2 = (1 - math.cos(math.sqrt(psi)))/ psi
                c_3 = (math.sqrt(psi) - math.sin(math.sqrt(psi))) / math.sqrt(psi**3)
            elif psi < -thresh:
                c_2 = (1 - math.cosh(math.sqrt(-psi))) / psi
                c_3 = (math.sinh(math.sqrt(-psi)) - math.sqrt(-psi)) / math.sqrt((-psi)**3)
            else:
                c_2 = 1 / 2
                c_3 = 1 / 6

        # compute delta_V transformation variables
        f = 1 - y / self.i_state.radius
        g_dot = 1 - y / self.f_state.radius
        g = A * math.sqrt(y / self.i_state.mu)

        # compute delta_v's
        dv_0 = (self.i_state.cart_pos() - f * self.f_state.cart_pos()) / g 
        dv_1 = (g_dot * self.f_state.cart_pos() - self.i_state.cart_pos()) / g

        return (dv_0, dv_f, delta_t)

    def __auto_dm(self):
        """Automatically selects direction of motion with shortest transfer"""
        if self.delta_true_anom < np.pi:
            return 1
        else:
            return -1

    def __get_y(self, A, psi, c_2, c_3):
        """Finds the variable y 
        
        Args: 
        A (float): "A" variable in sol-n algorithm (see paper cited in module docstring)
        psi (float): psi variable in sol-n algorithm (see paper cited in module docstring)
        c_2 (float): c_2 variable in sol-n algorithm (see paper cited in module docstring)
        c_3 (float): c_3 variable in sol-n algorithm (see paper cited in module docstring)

        Returns: 
        (float)
        """
        y = self.i_state.radius + self.f_state.radius + (A * (psi * c_3 - 1)) / math.sqrt(c_2)
        return y

    def __adjust_psi(self, A, psi_low, c_2, c_3):
        """Adjusts psi lower bound up to ensure y variable is greater than 0

        Args: 
        A (float): "A" variable in sol-n algorithm (see paper cited in module docstring)
        psi (float): psi variable in sol-n algorithm (see paper cited in module docstring)
        c_2 (float): c_2 variable in sol-n algorithm (see paper cited in module docstring)
        c_3 (float): c_3 variable in sol-n algorithm (see paper cited in module docstring)

        Returns: 
        (float)
        """
        y = self.__get_y(A, psi_low, c_2, c_3)
        while y < 0:
            psi_low += 0.1
            y = self.__get_y(A, psi_low, c_2, c_3)
        return psi_low

    def __get_delta_t(self, chi, c_3, A, y):
        """Finds the estimate transfer time for a given set of universal parameters

        Args: 
        chi (float): chi variable in sol-n algorithm (see paper cited in module docstring)
        c_2 (float): c_2 variable in sol-n algorithm (see paper cited in module docstring)
        c_3 (float): c_3 variable in sol-n algorithm (see paper cited in module docstring)
        y (float): y variable in sol-n algorithm (see paper cited in module docstring)
        
        Returns: 
        (float): transfer time (sec)
        """
        delta_t = chi**3 * c_3 + A * math.sqrt(y) / math.sqrt(self.i_state.mu)
        return delta_t
