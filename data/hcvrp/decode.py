import pickle

# Specify the path to your pickle file
pickle_file_path = 'data/hcvrp/hcvrp_v3_40_seed24610_tw.pkl'

# Load data from the pickle file
with open(pickle_file_path, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"Total instance: {len(loaded_data)}")
print(f"Get 0 instance: {loaded_data[0]}")
print()

instance_num = 0
depot_coordinates = loaded_data[instance_num][0]
customer_coordinates = loaded_data[instance_num][1]
customer_demands = loaded_data[instance_num][2]
vehicle_capacities = loaded_data[0][3]
time_open = loaded_data[0][4]
print(f"Depot coordinate location 1st instance: {depot_coordinates}")
print("")
print(f"Coordinate of customer 1st instance: {customer_coordinates}")
print(f"Num. of customers (coordinates): {len(customer_coordinates)}")
print("")
print(f"Demand of customer: {customer_demands}")
print(f"Num. of customers (demand): {len(customer_demands)}")
print("")
print(f"Capicity of 3 vehicle: {vehicle_capacities}")
print()
import numpy as np
print("whatever is this:", np.asarray(loaded_data[0][4]).shape)

import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(loaded_data)