#!/usr/bin/env python
"""Propagator Module

Author: Hunter Mellema
Summary: provides a class to propagate, plot, and export trajectories

"""
import numpy as np
from states import *
import pandas as pd

class PropagatorKeplerian(object):
    """ """
    def __init__(self, initial_state):
        if  isinstance(initial_state, Cartesian_State):
           self.initial_state = initial_state.to_kep_state()
        elif isinstance(initial_state, Keplerian_State):
            self.initial_state = initial_state
        else:
            raise TypeError("No valid type of state object was provided")

        self.prop_states = [self.initial_state]

    def propagate(self, end_time, time_step=60):
        """Propagates the orbit forward or backwards to a given time

        Args:
        end_time (float): time to propagate up to (seconds)
        time_step (float): propagation time step (seconds)

        Returns:
        (tuple(list[float])): radii, velocity, time lists
        """
        for time in np.arange(self.initial_state.time, end_time, time_step):
            self.prop_states.append(self.initial_state.propagate(time))

        radii = [s.radius for s in self.prop_states]
        velocities = [np.linalg.norm(s.cart_vel) for s in self.prop_states]
        times = [s.time for s in self.prop_states]

        return (radii, velocities, times)

    def export(self):
        """ Exports state data to csv """
        raise NotImplementedError("Export for Keplerian_State not supported yet")


class PropagatorCartesian(object):
    """Propagator object for cartesian states

    Args:

    initial_state (Cartesian_State): cartesian state object to use as initial condition

    """
    def __init__(self, initial_state):
        if  isinstance(initial_state, Cartesian_State):
           self.initial_state = initial_state
        else:
            raise TypeError("No valid type of state object was provided (Must be CartesianState)")

        self.prop_states = [self.initial_state]

    def propagate(self, end_time, time_step=10):
        """Propagates cartesian state forward or backwards in time

        Args:
        end_time (float): time to propagate orbit to (seconds)
        time_step (float): step between propagation states (seconds) [default = 10]

        Returns:
        (tuple(list[np.array(float)])): postion, velocties, and times for each state
        """
        state_cart = self.initial_state
        for time in np.arange(self.initial_state.time + time_step, end_time, time_step):
            state_cart = state_cart.propagate(time)
            self.prop_states.append(state_cart)

        positions = [s.pos for s in self.prop_states]
        velocities = [s.vel for s in self.prop_states]
        times = [s.time for s in self.prop_states]

        return (positions, velocities, times)

    def export_states(self, filename):
        """Exports data to a csv file

        Args:

        filename (str): name of file or filepath to export data to
        """
        x_pos, y_pos, z_pos, vxs, vys, vzs = ([] for i in range(6))
        for state in self.prop_states: 
            x, y, z = state.pos
            vx, vy, vz = state.vel
            x_pos.append(x)
            y_pos.append(y)
            z_pos.append(z)
            vxs.append(vx)
            vys.append(vy)
            vzs.append(vz)

        times = [s.time for s in self.prop_states]

        d = {'times': times, 'x_pos': x_pos, 'y_pos': y_pos, 'z_pos': z_pos,
             'vx': vxs, 'vy': vys, 'vz': vzs}
        df = pd.DataFrame(data=d)
        df.to_csv(filename, index_label='iters')

    def export_params(self, filename):
        """Exports data to a csv file

        Args:

        filename (str): name of file or filepath to export data to
        """
        radii, hs, incls, raans, eccs, arg_peris, true_anoms, nrgs = ([] for i in range(8))
        for state in self.prop_states: 
            temp_state = state.to_kep_state()
            radii.append(temp_state.radius)
            hs.append(temp_state.h)
            incls.append(temp_state.incl)
            raans.append(temp_state.raan)
            eccs.append(temp_state.ecc)
            arg_peris.append(temp_state.arg_peri)
            true_anoms.append(temp_state.true_anom)
            nrgs.append(temp_state.nrg)

        times = [s.time for s in self.prop_states]

        d = {'time': times, 'radius': radii, 'inclination': incls, 'RAAN': raans, 'Eccentricities': eccs,
             'Argument of Perigee': arg_peris, 'True anomaly': true_anoms, 'Energy': nrgs }
        df = pd.DataFrame(data=d)
        df.to_csv(filename, index_label='iters')

        return (times, radii, hs, incls, raans, eccs, arg_peris, true_anoms, nrgs)
