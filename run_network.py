# Import the classes as well as the global variables from class_definitions.py
import class_definitions
SONG_LENGTH, DT, SONG_TIME_POINTS = class_definitions.SONG_LENGTH, class_definitions.DT, class_definitions.SONG_TIME_POINTS
Layer, Bird = class_definitions.Layer, class_definitions.Bird

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import types

"""
Both HVC and LMAN need firing templates using their specific synaptic conductance values
Here is a function that can be added as a method to both the HVC and LMAN layers for that
"""
def generate_synaptic_template(self):
    # First, universal lambda function for calculating synaptic conductance at time t
    # t = time, cc = conductance const, ct = conductance tau, s = slow and f = fast
    g = lambda t, cc_s, ct_s, cc_f, ct_f : (cc_s * np.exp(t/-ct_s)) - (cc_f * np.exp(t/-ct_f))
    self.template = [
        g(t, self.synaptic_constants["slow_cond_const"], self.synaptic_constants["slow_cond_tau"], self.synaptic_constants["fast_cond_const"], self.synapyic_constants["fast_cond_tau"]) for t in SONG_TIME_POINTS
    ]
    return self

"""
Initiate RA layer
This includes defining HVC firing template method and adding to HVC instance
"""
NEURON_NUMBER_HVC = 100
HVC_INPUTS = 0
HVC_SYNAPTIC_CONSTANTS_DICT = {
    "slow_cond_const": 1.,
    "slow_cond_tau": 5.,
    "fast_cond_const": 1.,
    "fast_cond_tau": 1.,
    "last_event_buffer": 10 / DT # last HVC neuron should spike XXX ms before the end of the song
}
def HVC_output_generator(self):
    spike_timings = np.linspace(0, SONG_TIME_POINTS-self.synaptic_constants["last_event_buffer"], self.number_of_neurons)
    for row, spike in enumerate(spike_timings):
        self.outputs[row,int(spike):] = self.template[:int(spike)]
    return self

HVC = Layer(
    number_of_neurons=NEURON_NUMBER_HVC,
    number_of_inputs=HVC_INPUTS,
    synaptic_constants=HVC_SYNAPTIC_CONSTANTS_DICT
)

HVC.generate_template = types.MethodType( generate_synaptic_template, HVC )
HVC.generate_output = types.MethodType( HVC_output_generator, HVC )

"""
Initiate RA layer
This includes defining the specific RA conductance calculation method and adding it to the RA instance
"""
NEURON_NUMBER_RA = 10
RA_SYNAPTIC_CONSTANTS_DICT = {}
def RA_conductance_calc(self):
    pass

RA = Layer(
    number_of_neurons=NEURON_NUMBER_RA,
    number_of_inputs=NEURON_NUMBER_HVC,
    synaptic_constants=RA_SYNAPTIC_CONSTANTS_DICT
)

RA.conductance_calculation = types.MethodType( RA_conductance_calc, RA )


def run_network():
    NUMBER_OF_SONGS = 1000
    # Before any song production, intiate inputs/outputs for HVC and RA
    HVC.initiate_outputs()
    HVC.generate_template()
    HVC.generate_output()

    RA.generage_synaptic_weights()
    RA.initiate_outputs()

    for song in NUMBER_OF_SONGS:
        Bird.sing()

if __name__ == "__main__":
    run_network()