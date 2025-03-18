import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from argparse import ArgumentParser
import json

def main(args):
    initial_seed = 42

    n =  initial_seed
    seed = initial_seed
    sim = args.case_json

    # SETUP SIMULATION

    input_reader = InputReader(sim, "numerical_setup_forced_LES.json")
    initializer  = Initializer(input_reader)

    sim_manager  = SimulationManager(input_reader)

    # RUN SIMULATION
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--resume", action='store_true')
     parser.add_argument("--case_json", type=str)
     args = parser.parse_args()
     main(args)