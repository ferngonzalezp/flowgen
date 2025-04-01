import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
import json
from argparse import ArgumentParser
import os
import shutil
from mpi4py import MPI
import jax

# Initialize JAX distributed execution
jax.distributed.initialize()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_latest_directory(parent_dir, prefix):
    """Find the latest created directory in the parent directory that starts with a given prefix."""
    try:
        # Get all subdirectories that start with the given prefix
        subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                   if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(prefix)]
        if not subdirs:
            return None
        # Return the latest directory based on creation time
        return max(subdirs, key=os.path.getctime)
    except Exception as e:
        print(f"Error finding latest directory with prefix '{prefix}': {e}")
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

def get_available_gpus():
    """Detect available GPUs on the node."""
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    return num_gpus if num_gpus > 0 else 1  # Assume at least 1 GPU if not set

def main(args):
    print(f"Rank {rank}: Starting execution")
    devices = jax.devices()  # Get available GPUs
    num_devices = jax.device_count()
    
    # Each rank gets a GPU assigned
    gpu_id = rank % num_devices  

    device = jax.local_devices()  # Assign GPU

    # Assign a unique seed for each process
    seed = [args.initial_seed + size] * size
    realizations = np.inf if args.realizations == -1 else args.realizations

    # Each process picks a simulation case
    sim_cases = ["HIT_decay_ma0.2.json", "HIT_decay_ma0.4.json", "HIT_decay_ma0.6.json"]
    sim_case = sim_cases[rank % len(sim_cases)]  # Assign case based on rank

    n =  [1] * size
    if args.resume:
        latest_dir = get_latest_directory("args.sim_directory", f"sim_{rank}")

        sim_params = os.path.join(latest_dir, f'sim_{rank}.json')
        numerical_setup = os.path.join(latest_dir, 'numerical_setup.json')
        

        with open(sim_params, 'r') as json_file:
                resume_sim = json.load(json_file)
        
        seed[rank] = resume_sim['initial_condition']['turb_init_params']['seed']
        n[rank] = seed[rank] // rank

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
        comm.Barrier()

        n[rank] += 1
        clean_directory(parent_dir=args.sim_directory, dirname=f'sim_{rank}')

    
    while n[rank] <= realizations:
            #comm.Barrier()
            # SETUP SIMULATION
            # Modify simulation parameters
            sim_case = sim_cases[n[rank] % len(sim_cases)]
            with open(sim_case, 'r') as json_file:
                modified_sim = json.load(json_file)
            with open("numerical_setup_stream.json", 'r') as json_file:
                numerical_setup = json.load(json_file)
            
            modified_sim['initial_condition']['turb_init_params']['seed'] = seed[rank]
            modified_sim['general']['case_name']= f"sim_{rank}"
            modified_sim['general']['save_path'] = args.sim_directory

            # Save modified parameters
            with open(f"sim_{rank}.json", 'w') as json_file:
                json.dump(modified_sim, json_file, indent=4)

            # Run simulation
            print(f"Rank {rank}: Running simulation {n[rank]} with seed {seed[rank]}")
            input_reader = InputReader(f"sim_{rank}.json", "numerical_setup_stream.json")
            initializer = Initializer(input_reader)
            sim_manager = SimulationManager(input_reader)

            buffer_dictionary = initializer.initialization()
            sim_manager.simulate(buffer_dictionary)

            n[rank] += 1
            seed[rank] += size  # Ensure unique seed across ranks
            print(f"Rank {rank}: Cleaning up directory: sim_{rank}")
            clean_directory(parent_dir=args.sim_directory, dirname=f"sim_{rank}")
            #comm.Barrier()
        
if __name__ == "__main__":
     parser = ArgumentParser()
     parser.add_argument("--realizations", type=int, default=-1)
     parser.add_argument("--initial_seed", type=int, default=10)
     parser.add_argument("--sim_directory", type=str, default='./train_online')
     parser.add_argument("--resume", action='store_true')
     args = parser.parse_args()
     main(args)