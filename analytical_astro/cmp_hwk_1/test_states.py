#!/usr/bin/env python
"""
Name: test_states
Summary Test script for states module  

Author: Hunter Mellema
Notes: Written to run with pytest framework

"""
import pytest
from .states import * 

@pytest.fixture
def ref_cart_case_circ():
    """ A reference cartesian state for a circular orbit of 7000km radius"""
    pos = np.array([7000, 0, 0]) #km 
    vel = np.array([0, 7.456, 0]) #km 
    mu = 398600
    time = 0
    return Cartesian_State(pos, vel, mu, time)

@pytest.fixture
def ref_cart_case_1():
    """ Cartesian state reference case 
    I use the example state from example 4.3 in Curtis ("Orbit mechanics for 
    engineering students") to check against
    """
    pos = np.array([-6045, -3490, 2500]) #km 
    vel = np.array([-3.457, 6.618, 2.533]) #km
    mu = 398600
    time = 0
    return Cartesian_State(pos, vel, mu, time)

class TestCartesianStateBaseMethods(object):
    """ Ensures proper initialization and operation of cartesian state object """

    def test_h(self, ref_cart_case_1, ref_cart_case_circ):
        """ Tests that h initializes properly """
        correct_hs = (58310, 52192)
        ref_states = (ref_cart_case_1, ref_cart_case_circ)
        for correct_h, ref_state in zip(correct_hs, ref_states):
            assert(ref_state.h == pytest.approx(correct_h, 1))

    def test_nrg(self, ref_cart_case_1, ref_cart_case_circ):
        """ Tests that energy attribute is properly calculated """
        correct_nrgs = (-22.6786, -28.4714)
        ref_states = (ref_cart_case_1, ref_cart_case_circ)
        for correct_nrg, ref_state in zip(correct_nrgs, ref_states):
            assert(ref_state.nrg == pytest.approx(correct_nrg, 1))

    def test_case_1_to_kep_state(self, ref_cart_case_1):
        """ Checks that the case 1 state is correctly converted to a keplerian state """
        correct_values = {'h': 58310,
                          'incl': 2.6738,
                          'raan': 4.4558256,
                          'ecc': 0.1712,
                          'arg_peri': .35028758,
                          'true_anom': .49653617,
                          'a': 8788,
                          'mu': 398600,
                          'time': 0}
        kep_state = ref_cart_case_1.to_kep_state()
        # check all attributes of keplerian state
        assert(kep_state.h == pytest.approx(correct_values['h'],1))
        assert(kep_state.incl == pytest.approx(correct_values['incl'], 0.01))
        assert(kep_state.raan == pytest.approx(correct_values['raan'], 0.01))
        assert(kep_state.ecc == pytest.approx(correct_values['ecc'], 0.01))
        assert(kep_state.arg_peri == pytest.approx(correct_values['arg_peri'], 0.01))
        assert(kep_state.true_anom == pytest.approx(correct_values['true_anom'], 0.01))
        assert(kep_state.a == pytest.approx(correct_values['a'],0.1))
        assert(kep_state.mu == pytest.approx(correct_values['mu']))
        assert(kep_state.time == pytest.approx(correct_values['time']))
        
    def test_circ_to_kep_state(sefl, ref_cart_case_circ):
        """ Checks that the circular orbit state is correctly converted to a keplerian state"""
        correct_values = {'h': 58310,
                          'incl': 0,
                          'raan': np.nan,
                          'ecc': 0.01,
                          'arg_peri': np.nan,
                          'true_anom': np.nan,
                          'a': 7000,
                          'mu': 398600,
                          'time': 0}
        kep_state = ref_cart_case_circ.to_kep_state()
        # check all attributes of keplerian state
        assert(kep_state.h == pytest.approx(correct_values['h'],1))
        assert(kep_state.incl == pytest.approx(correct_values['incl'],0.1))
        assert(np.isnan(kep_state.raan))
        assert(kep_state.ecc == pytest.approx(correct_values['ecc'],2))
        assert(np.isnan(correct_values['arg_peri']))
        assert(np.isnan(kep_state.true_anom))
        assert(kep_state.a == pytest.approx(correct_values['a'], .1))
        assert(kep_state.mu == pytest.approx(correct_values['mu']))
        assert(kep_state.time == correct_values['time'])

##### BEGIN KEPLERIAN TESTS ######
@pytest.fixture
def ref_kep_case_circ():
    """ A reference keplerian state for a circular orbit of 7000km radius"""
    mu = 398600
    time = 0
    return Keplerian_State(7000, 58310, 0, 0, 0, 0, 0, mu, time)

@pytest.fixture
def ref_kep_case_1():
    """ Keplerian test reference case 
    I use the example state from example 4.3 in Curtis ("Orbit mechanics for 
    engineering students") to check against
    """
    mu = 398600
    time = 0
    return Keplerian_State(7414, 58310, 2.6738, 4.455825, 0.1712,
                           .35028758, .49653617, mu, time)
    
class TestKeplerianStateBaseMethods(object):
    """ Ensures proper initialization and operation of Keplerian state object """


    def test_cart_pos(self, ref_kep_case_1):
        """ Tests that calculation of cartesian positions works properly """ 
        correct_pos = np.array([-6045, -3490, 2500])
        np.isclose(ref_kep_case_1.cart_pos,
                   correct_pos, rtol=.1)
        
    def test_cart_vel(self, ref_kep_case_1):
        """ Tests that calculation of cartesian velocities works properly """
        correct_vel = np.array([-3.457, 6.618, 2.533])
        np.testing.assert_array_almost_equal(ref_kep_case_1.cart_vel,
                                             correct_vel, decimal=1)

    def test_propagation(self, ref_kep_case_1):
        """ Tests that propagation functions properly 
        I solved this one by hand, using wolfram alpha to solve for the 
        eccentric anomaly
        """
        correct_values = {'radius': 7382,
                          'h': 58310,
                          'incl': 2.6738,
                          'raan': 4.455825,
                          'ecc': 0.1712,
                          'arg_peri': .35028758,
                          'true_anom': 0.43,
                          'a': 8788,
                          'mu': 398600,
                          'time': 100}
        kep_state = ref_kep_case_1.propagate(100)
        assert(kep_state.radius == pytest.approx(correct_values['radius'],.001))
        assert(kep_state.h == pytest.approx(correct_values['h'],1))
        assert(kep_state.incl == pytest.approx(correct_values['incl'], 0.01))
        assert(kep_state.raan == pytest.approx(correct_values['raan'], 0.01))
        assert(kep_state.ecc == pytest.approx(correct_values['ecc'], 0.01))
        assert(kep_state.arg_peri == pytest.approx(correct_values['arg_peri'], 0.01))
        assert(kep_state.true_anom == pytest.approx(correct_values['true_anom'], 0.01))
        assert(kep_state.a == pytest.approx(correct_values['a'],0.1))
        assert(kep_state.mu == pytest.approx(correct_values['mu']))
        assert(kep_state.time == pytest.approx(correct_values['time']))
