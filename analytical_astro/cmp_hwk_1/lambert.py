#!/usr/bin/env python
"""Lambert Module 

Author: Hunter Mellema
Summary: Provides a lambert_solver object that takes in 2 Keplerian_State objects and
a transfer time and calculates a family of possible transfer solutions 

Note: the first algorithm used here is taken from "Lambert's Problem" by Kate Davis from CCAR 
Note: The second algorithm used here is taken from the Paper "Revisiting Labert's Problem By Dario Izzo" 

"""
import numpy as np
import math

class LambertSolver(object):
    """
    """
    def __init__(self, initial_state, final_state, max_transfer_time, min_transfer_time=None):
        if not isinstance(initial_state, 'Keplerian_State') or not isinstance(final_state, 'Keplerian_State'):
            raise TypeError("The Lambert Solver can only accept Keplerian_State objects")
        if not initial_state.mu == final_state.mu:
            raise ValueError("Both states MUST have the same gravitaional parameter: mu")
        
        self.i_state = initial_state
        self.f_state = final_state

    def solve(self, delta_t_0, direction_of_motion=0, thresh=1 * 10**-6):
        """ """
        delta_true_anom = self.final_state.true_anom - self.i_state.true_anom
        A = self.__get_A(delta_true_anom, direction_of_motion)

        # set initial guesses 
        c2 = 1 / 2
        c3 = 1 / 6
        psi = 0
        psi_up = 4 * np.pi
        psi_low = -4 * np.pi

        delta_t = self.__get_delta_t(chi, c_3, A, y)
        
        while math.fabs(delta_t - delta_t_0) > thresh:

            y = self.__get_(A, psi, c_2, c_3)

            if A > 0 and y < 0:
                psi_low = self.__adjust_psi(A, psi, c_2, c_3)
                y = self.__get_(A, psi_low, c_2, c_3)
                
            chi = math.sqrt(y / c_2)
            delta_t = self.__get_delta_t(chi, c_3, A, y)

            if delta_t <= delta_t_0:
                psi_low = psi
            else:
                psi_up = psi

            # bisect psi
            psi = (psi_up + psi_low) / 2

            if psi > thresh:
                c_2 = (1 - math.cos(math.sqrt(psi)))/ psi
                c_3 = (math.sqrt(psi) - math.sin(math.sqrt(psi))) / math.sqrt(psi**3)
            elif psi < -thresh:
                c_2 = (1 - math.cosh(math.sqrt(-psi))) / psi 
                c_3 = (math.sinh(-psi) - math.sqrt(-psi)) / math.sqrt((-psi)**3)
            else:
                c_2 = 1 / 2
                c_3 = 1 / 6
                
        f = 1 - y / self.i_state.radius
        g_dot = 1 - y / self.f_state.radius
        g = A * math.sqrt(y / self.i_state.mu)
        
        return (dv_0, dv_f, delta_t)
        
    def __get_A(self, delta_true_anom, direction_of_motion):
        """ """ 
        A = math.sqrt(self.i_state.radius * self.f_state.radius * (1 + cos(delta_true_anom)))

        if A == 0 or delta_ecc_anom == 0:
            raise ValueError("A and delta_true_anom must NOT = 0")
                
        # change direction_of_motion
        if direction_of_motion == 0:
            if delta_true_anom < np.pi:
                pass
            else:
                A *= -1
        else: 
            A *= direction_of_motion
                
        return A 

                
    def __get_y(self, A, psi, c_2, c_3):
        """ """ 
        y = self.i_state.radius + self.f_state.radius + (A * (psi * c_3 - 1)) / math.sqrt(c_2)
        return y

                
    def __adjust_psi(self, A, psi, c_2, c_3):
        """ """ 
        y = self.__get_y(A, psi, c_2, c_3)
        while y < 0:
            psi = psi + 0.1
            y = self.__get_y(A, psi, c_2, c_3)
        return psi 
                

    def __get_delta_t(self, chi, c_3, A, y):
        """ """
        delta_t = chi**3 * c_3 + A * math.sqrt(y) / math.sqrt(self.i_state.mu)
        return delta_t
