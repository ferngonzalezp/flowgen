import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import json
from argparse import ArgumentParser
import os
import shutil

def get_latest_directory(parent_dir):
    """Find the latest created directory in the parent directory."""
    try:
        # Get all subdirectories
        subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                  if os.path.isdir(os.path.join(parent_dir, d))]
        if not subdirs:
            return None
        # Return the latest directory based on creation time
        return max(subdirs, key=os.path.getctime)
    except Exception as e:
        print(f"Error finding latest directory: {e}")
        return None

def clean_directory(parent_dir=".", dirname='HIT_'):
    """
    Delete all directories starting with 'HIT_' in the specified parent directory,
    except for the most recently created one.
    
    Args:
        parent_dir (str): Path to the parent directory (defaults to current directory)
        dirname (str): Prefix of directories to clean (defaults to 'HIT_')
    """
    try:
        # Get all directories starting with dirname
        hit_dirs = [d for d in os.listdir(parent_dir) 
                   if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(dirname)]
        
        if not hit_dirs:
            return
        
        # Get the full paths and sort by creation time
        hit_dirs_with_time = [(d, os.path.getctime(os.path.join(parent_dir, d))) 
                             for d in hit_dirs]
        hit_dirs_with_time.sort(key=lambda x: x[1])  # Sort by creation time
        
        # Keep the last directory, delete all others
        dirs_to_delete = hit_dirs_with_time[:-1]  # All except the last one
        
        # Delete directories
        for dir_name, _ in dirs_to_delete:
            dir_path = os.path.join(parent_dir, dir_name)
            try:
                shutil.rmtree(dir_path)
                print(f"Deleted: {dir_path}")
            except Exception as e:
                print(f"Error deleting {dir_path}: {e}")
                
        print(f"Successfully deleted {len(dirs_to_delete)} directories")
        print(f"Kept most recent directory: {hit_dirs_with_time[-1][0]}")
    except:
        pass

def main(args):
    if args.realizations == -1:
        realizations = np.inf
    else:
        realizations = args.realizations

    n =  1
    seed = args.initial_seed
    if args.resume:
        latest_dir = get_latest_directory("./train_online")

        sim_params = os.path.join(latest_dir, 'HIT.json')
        numerical_setup = os.path.join(latest_dir, 'numerical_setup.json')
        

        with open(sim_params, 'r') as json_file:
                resume_sim = json.load(json_file)
        
        seed = resume_sim['initial_condition']['turb_init_params']['seed']
        n = seed

        try:
            for root, dirs, files in os.walk(latest_dir):
                if "rst.h5" in files:
                    restart_sol = os.path.join(root, "rst.h5")
            if restart_sol:
                resume_sim['restart']['flag'] =  true
                resume_sim['restart']['file_path'] =  restart_sol
        except:
            pass

        with open(sim_params, 'w') as json_file:
                json.dump(resume_sim,json_file, indent=4)

        input_reader = InputReader(sim_params, numerical_setup)
        initializer  = Initializer(input_reader)

        sim_manager  = SimulationManager(input_reader)

        # RUN SIMULATION
        buffer_dictionary = initializer.initialization()
        sim_manager.simulate(buffer_dictionary)
        n += 1
        clean_directory(parent_dir='./train_online', dirname='HIT')

    
    while n <= realizations:

        for i, sim in enumerate(["HIT_decay_ma0.2.json",
                            "HIT_decay_ma0.4.json", 
                            "HIT_decay_ma0.6.json",
                            ]):
            # SETUP SIMULATION
            #seed += i  
            with open(sim, 'r') as json_file:
                modified_sim = json.load(json_file)
            with open("numerical_setup_stream.json", 'r') as json_file:
                numerical_setup = json.load(json_file)
            
            modified_sim['initial_condition']['turb_init_params']['seed']=seed

            with open(sim, 'w') as json_file:
                json.dump(modified_sim,json_file, indent=4)

            input_reader = InputReader(sim, "numerical_setup_stream.json")
            initializer  = Initializer(input_reader)

            sim_manager  = SimulationManager(input_reader)

            # RUN SIMULATION
            buffer_dictionary = initializer.initialization()
            sim_manager.simulate(buffer_dictionary)
            clean_directory(parent_dir='./train_online', dirname='HIT')
        
        n += 1
        seed += 1 
        
if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--realizations", type=int, default=-1)
     parser.add_argument("--initial_seed", type=int, default=10)
     parser.add_argument("--resume", action='store_true')
     args = parser.parse_args()
     main(args)