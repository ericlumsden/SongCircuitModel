from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import types

"""
v3 of the layer class model now has distinct 'layer' classes that keep track of the following basics:
 - synaptic weights
 - layer-specific constants (e.g. conductance values, time constants)

Additionally, there is a 'bird' layer that performs the song using the circuit elements from the layer classes themselves
This was necessary as the following will all be simultaneously calculated:
 - DON'T FORGET ABOUT TIME DELAY IN THE ERROR CALCULATIONS, THE ERROR IS CALCULATED AS @CURRENT TIME POINT BUT APPLIES TO DW T_DELAY IN THE PAST... talking with David, is this necessary? The decay in the eligibility should account for any remaining effects from past events... the 'critic' is essentially taking in the info @real-time...
 - RA neuron activity - inhibition need the number of neurons that spiked on the previous time step (* 2/n_neurons)
 - MP neuron activity
 - Song replication error
 - Eligibility - Need to keep previous 5 LMAN activity to average
 - Running dW matrix

Then, at the end of each song:
 - MP neurons concatenate their latest output to their running average matrix after dropping the 5th most recent output
 - LMAN layer needs to concatenate latest activity to running average matrix as well
 - RA synaptic weights are updated by dW
"""

# Universal constants
SONG_LENGTH = 1000. # ms
DT = 0.5 # ms
SONG_TIME_POINTS = int(SONG_LENGTH / DT)

""" 
The 'Layer' class will be used for the primary song circuit nuclei (HVC, RA, MNs, LMAN) 
Layer specific methods will be added to the individual instances of the class post hoc
"""
@dataclass
class Layer:
    number_of_neurons: int
    number_of_inputs: int
    synaptic_constants: dict
    outputs = np.array([])
    synaptic_weights = np.array([])
    outputs = np.array([])

    def generage_synaptic_weights(self):
        self.synaptic_weights = np.random.randn((self.number_of_neurons, self.number_of_inputs))
        return self
    
    def initiate_outputs(self):
        self.outputs = np.zeros((self.number_of_neurons, SONG_TIME_POINTS))
        return self
    
    def synaptic_conductance_template(self):
        # This needs to be used for both HVC and LMAN layers to generate their firing patterns
        return self

    def plot(self, x, y, label_dict):
        # Plot whatever aspect of the layer gets passed to the function
        plt.plot(x, y)
        plt.xlabel(label_dict["xlabel"])
        plt.ylabel(label_dict["ylabel"])
        plt.title(label_dict["title"])
        plt.savefig(f"./{type(self).__name__}_{label_dict['save_file_name']}.png")
        return self


@dataclass
class Bird:
    motor_out_template_1: np.array([])
    motor_out_template_2: np.array([])

    def calculate_RA_synaptic_conductance(self, RA, HVC_RA_projections, LMAN_projections, time_step):
        next_time_step = time_step + 1
        return self
    
    def calculate_MN_outputs(self):
        return self

    def sing(self, HVC, RA, LMAN, time_points=SONG_TIME_POINTS):
        dW = np.zeros((RA.number_of_neurons, HVC.number_of_neurons))
        HVC_RA_projections = RA.synaptic_weights @ HVC.outputs # matmul shorthand
        LMAN.generate_outputs()
        """ With each song bout calculate the following @tn:
        - RA inputs received
        - RA total activity for inhibitory conductances from prior time steps' firing
        - RA outputs
        - Motor pool inputs and outputs
        - Error from template
            - Error is calculated on 25ms delay ????
            - Skip AT LEAST first 50 time steps then begin Error, Eligibility and dW calculations
            - ...(check when RA activity ramps up to baseline, that will dictate 'centerfold')
        - Eligibility
        - Update the dW matrix
        """
        for t in time_points:
            # Do the above at each time point
            self.calculate_RA_synaptic_conductance(RA, HVC_RA_projections, LMAN.outputs, t)
            self.calculate_MN_outputs()
            continue
        # Once through the song, update synaptic weights of RA, update MP neuron running average
        RA.synaptic_weights += dW
        return self

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
    "last_event_buffer": 10 / DT # last HVC neuron sohuld spike XXX ms before the end of the song
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