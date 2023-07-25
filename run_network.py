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
    spike_train_array = np.zeros(int(SONG_TIME_POINTS))
    np.put(spike_train_array, [int(spike * self.synaptic_constants["spike_train_isi"] / DT) for spike in range(self.synaptic_consants["spikes_per_train"])], [1.]*self.synaptic_constants["spikes_per_train"])
    single_event_template = [
        g(t, self.synaptic_constants["slow_cond_const"], self.synaptic_constants["slow_cond_tau"], self.synaptic_constants["fast_cond_const"], self.synapyic_constants["fast_cond_tau"]) for t in SONG_TIME_POINTS
    ]
    train_template = np.convolve(spike_train_array, single_event_template)
    self.template = train_template[:int(SONG_TIME_POINTS)]
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
    "last_event_buffer": 10 / DT, # last HVC neuron should spike XXX ms before the end of the song
    "spike_train_isi": 2, # ms between spikes in spike train
    "spikes_per_train": 3
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
This includes defining the specific RA conductance calculation method and leaky-integrate-and-fire method and adding them to the RA instance
"""
NEURON_NUMBER_RA = 10
RA_SYNAPTIC_CONSTANTS_DICT = {}

def RA_leaky_integrate_fire(self, t):
    pass

def RA_conductance_calc(self, t):
    if t == 0: # Base case
        pass
    else:
        return self

RA = Layer(
    number_of_neurons=NEURON_NUMBER_RA,
    number_of_inputs=NEURON_NUMBER_HVC,
    synaptic_constants=RA_SYNAPTIC_CONSTANTS_DICT
)

RA.conductance_calculation = types.MethodType( RA_conductance_calc, RA )
RA.leaky_integrate_and_fire = types.MethodType( RA_leaky_integrate_fire, RA )


"""
Initiate motor neuron layer
"""
NEURON_NUMBER_MN = 2
MN_synaptic_dict = {}

MotorPool = Layer(
    number_of_neurons=NEURON_NUMBER_MN,
    number_of_inputs=RA.number_of_neurons,
    synaptic_constants=MN_synaptic_dict
)


"""
Initiate LMAN
LMAN has its own firing properites/pattern, so functions must be defined and added as methods to the LMAN layer instance
"""
NEURON_NUMBER_LMAN = 1
LMAN_synaptic_dict = {
    "slow_cond_const": 1.,
    "slow_cond_tau": 5.,
    "fast_cond_const": 1.,
    "fast_cond_tau": 1.,
    "firing_rate": 0.08, # 80 Hz, in ms
    "spike_train_isi": 0,
    "spikes_per_train": 1
}

def LMAN_output_generator(self):
    outputs_random_template = np.random.rand(self.number_of_neurons, int(SONG_TIME_POINTS))
    spike_times = ((self.synaptic_constants["firing_rate"]) * DT) > outputs_random_template
    for neuron in range(self.number_of_neurons):
        temp_row = np.convolve(spike_times[neuron, :], self.conductance_template)
        self.outputs[neuron, :] = temp_row[:int(SONG_TIME_POINTS)]
    return self

LMAN = Layer(
    number_of_neurons=NEURON_NUMBER_LMAN,
    number_of_inputs=MotorPool.number_of_neurons,
    synaptic_constants=LMAN_synaptic_dict
)

LMAN.generate_template = types.MethodType( generate_synaptic_template, LMAN )
LMAN.generate_output = types.MethodType( LMAN_output_generator, LMAN )


# Here's a function that will actually run the program itself, called with __name__ == "__main__"
def run_network():
    NUMBER_OF_SONGS = 1000
    # Before any song production, intiate inputs/outputs matrices for HVC, RA and LMAN and template for synaptic events
    HVC.initiate_outputs()
    HVC.generate_template()
    HVC.generate_output()
    RA.generage_synaptic_weights()
    RA.initiate_outputs()
    LMAN.initiate_outputs()
    LMAN.generate_template()

    for song in NUMBER_OF_SONGS:
        LMAN.generate_output()
        Bird.sing(HVC=HVC, RA=RA, LMAN=LMAN)


if __name__ == "__main__":
    run_network()