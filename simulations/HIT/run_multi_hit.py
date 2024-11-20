import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import json
realizations = np.inf
initial_seed = 10

n =  initial_seed
seed = initial_seed
while True:
    for i, sim in enumerate(["HIT_decay_ma0.2.json",
                          "HIT_decay_ma0.4.json", 
                          "HIT_decay_ma0.6.json",
                          ]):
        # SETUP SIMULATION
        seed += i  
        with open(sim, 'r') as json_file:
            modified_sim = json.load(json_file)

        modified_sim['initial_condition']['turb_init_params']['seed']=seed

        with open(sim, 'w') as json_file:
            json.dump(modified_sim,json_file, indent=4)

        input_reader = InputReader(sim, "numerical_setup.json")
        initializer  = Initializer(input_reader)

        sim_manager  = SimulationManager(input_reader)

        # RUN SIMULATION
        buffer_dictionary = initializer.initialization()
        sim_manager.simulate(buffer_dictionary)
    
    n += 1
    seed += 1    
    if n >= realizations:
        break