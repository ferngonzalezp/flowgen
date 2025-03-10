import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import json
initial_seed = 42

n =  initial_seed
seed = initial_seed
sim = "/scratch/cfd/gonzalez/flowgen/simulations/HIT/HIT_decay_ma1.2.json"

# SETUP SIMULATION

input_reader = InputReader(sim, "numerical_setup_forced.json")
initializer  = Initializer(input_reader)

sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)