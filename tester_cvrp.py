import time
import numpy as np
from iter_solver import sisr_cvrp

def parse_vrp_question(file_path):
    data = []
    vehicle_capcity = -1
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            sline = line.strip()
            # print(i, "---", sline)
            if len(sline)==0: continue
            if i==3:
                vehicle_capcity = int(sline.split()[1])
            if i>=6:
                data_to_append = [int(i) for i in sline.split()][1:]
                data.append(data_to_append)
                # print("Append Data:", data_to_append)
            i+=1
    return vehicle_capcity, np.array(data)

# vehicle_capcity, data = parse_vrp_question("data/homberger_200_customer_instances/C1_2_1.TXT")
vehicle_capcity, data = parse_vrp_question("data/sample_50.txt")
vehicle_capcity, data = parse_vrp_question("data/sample_10.txt")
vehicle_capcity, data = parse_vrp_question("data/sample_100.txt")
vehicle_capcity, data = parse_vrp_question("data/sample.txt")
vehicle_capcity, data = parse_vrp_question("data/sample_20.txt")
# Columns: XCOORD. | YCOORD. | DEMAND | READY TIME | DUE DATE | SERVICE TIME
data = data[:, :3]
# Columns: XCOORD. | YCOORD. | DEMAND

print("Vehicle Capacity :", vehicle_capcity)
print("data.shape       :", data.shape)
print("------------------")

start_time = time.time()
np.random.seed(0)
d, best_routes = sisr_cvrp(data, vehicle_capcity, n_iter=10_000, init_T=10.0, final_T=1.0,
                           init_route = None, verbose_step=50)
time_cost = time.time() - start_time
print("time_cost", time_cost)
print("distance", d)