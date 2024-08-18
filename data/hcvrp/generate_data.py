import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import check_extension, save_dataset
import torch
import pickle
import argparse

def generate_hcvrp_data(dataset_size, hcvrp_size, veh_num):
    data = []
    for seed in range(24601, 24611):
        rnd = np.random.RandomState(seed)

        loc = rnd.uniform(0, 1, size=(dataset_size, hcvrp_size + 1, 2))
        depot = loc[:, -1]
        cust = loc[:, :-1]
        d = rnd.randint(1, 10, [dataset_size, hcvrp_size+1])
        d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here

        ## Start time window
         # Initialize time windows with zeros
        time_windows = np.zeros((dataset_size, hcvrp_size, 24))

        # Define shifts
        shift_1 = range(7, 12)  # 7 AM to 12 PM
        shift_2 = range(13, 18)  # 1 PM to 6 PM

        # Randomly assign shifts to each customer
        for i in range(dataset_size):
            for j in range(hcvrp_size):
                shift_choice = rnd.choice(['shift_1', 'shift_2', 'both'])
                if shift_choice == 'shift_1':
                    time_windows[i, j, shift_1] = 1
                elif shift_choice == 'shift_2':
                    time_windows[i, j, shift_2] = 1
                elif shift_choice == 'both':
                    time_windows[i, j, shift_1] = 1
                    time_windows[i, j, shift_2] = 1
        ## END Tme window

        if veh_num == 3:
            cap = [20., 25., 30.]
            thedata = list(zip(depot.tolist(),  # Depot location
                               cust.tolist(),
                               d.tolist(),
                               np.full((dataset_size, 3), cap).tolist(),
                               time_windows.tolist()  # Time windows with shifts
                                ))
            data.extend(thedata)

        elif veh_num == 5:
            cap = [20., 25., 30., 35., 40.]
            thedata = list(zip(depot.tolist(),  # Depot location
                               cust.tolist(),
                               d.tolist(),
                               np.full((dataset_size, 5), cap).tolist()
                               ))
            data.append(thedata)

    print(f"data {data[0]}")
    print(f"total data {len(data)}")
    # print(f"data size {data.size}")
    # data = np.array(data).reshape(1280, 4)
    print(f"total data {len(data)}")
    return data

def visualize_hcvrp_data(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, instance in enumerate(data):
        depot, customers, demands, capacities, _ = instance
        num_customers = len(customers)
        
        # Convert customers to a NumPy array
        customers = np.array(customers)
        
        # Create a scatter plot to visualize depot and customer locations
        plt.figure(figsize=(8, 6))
        plt.scatter(depot[0], depot[1], c='g', marker='s', label='Depot')
        plt.scatter(customers[:, 0], customers[:, 1], c='b', marker='o', label='Customers')
        
        # Add labels for customer indices
        for j, (x, y) in enumerate(customers):
            plt.annotate(str(j + 1), (x, y), fontsize=12)
        
        plt.title(f'HCVRP Instance {i+1}')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.legend()
        
        # Save the visualization as an image (e.g., JPG)
        image_filename = os.path.join(save_dir, f'hcvrp_instance_{i+1}.jpg')
        plt.savefig(image_filename)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--dataset_size", type=int, default=128, help="1/10 Size of the dataset")
    parser.add_argument("--veh_num", type=int, default=3, help="number of the vehicles; 3 or 5")
    parser.add_argument('--graph_size', type=int, default=40,
                        help="Sizes of problem instances: {40, 60, 80, 100, 120} for 3 vehicles, "
                             "{80, 100, 120, 140, 160} for 5 vehicles")

    opts = parser.parse_args()
    data_dir = 'data'
    problem = 'hcvrp'
    datadir = os.path.join(data_dir, problem)
    os.makedirs(datadir, exist_ok=True)
    seed = 24610  # the last seed used for generating HCVRP data
    np.random.seed(seed)
    print(opts.dataset_size, opts.graph_size)
    filename = os.path.join(datadir, '{}_v{}_{}_seed{}_tw.pkl'.format(problem, opts.veh_num, opts.graph_size, seed))

    dataset = generate_hcvrp_data(opts.dataset_size, opts.graph_size, opts.veh_num)
    print(dataset[0])
    save_dataset(dataset, filename)
    visualize_hcvrp_data(dataset, os.path.join(data_dir, 'visualization'))


