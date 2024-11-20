import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import json
initial_seed = 42

n =  initial_seed
seed = initial_seed
sim = "HIT_forced_ma0.4.json"

# SETUP SIMULATION

input_reader = InputReader(sim, "forced_numerical_setup.json")
initializer  = Initializer(input_reader)

sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)