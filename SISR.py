import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import pandas as pd

class SISR(object):
    def __init__(
        self, 
        data: pd.DataFrame,
        vehicle_capacities: list[float],
        n_iter: int,
        n_iter_fleet: int = None,
        fleet_gap: int = None,
        init_T: float = 100.0,
        final_T: float = 1.0,
        c_bar: float = 10.0,
        L_max: float = 10.0,
        m_alpha: float = 0.01,
        obj_n_routes: int = None,
        init_route: list = None,
        verbose_step: int = None,
        test_obj: float = None,
        time_window: bool = False,
        soft_time_window: bool = False,
        multi_trip: bool = True,
        max_hour_per_vehicle: int = 12,
    ):
        self.data = data
        self.vehicle_capacities = vehicle_capacities
        self.vehicle_nums = len(vehicle_capacities)
        self.n_iter = n_iter
        self.n_iter_fleet = n_iter_fleet
        self.fleet_gap = fleet_gap
        self.init_T = init_T
        self.final_T = final_T
        self.c_bar = c_bar
        self.L_max = L_max
        self.m_alpha = m_alpha
        self.obj_n_routes = obj_n_routes
        self.verbose_step = verbose_step
        self.test_obj = test_obj
        self.time_window = time_window
        self.soft_time_window = soft_time_window
        self.multi_trip = multi_trip
        self.max_hour_per_vehicle = max_hour_per_vehicle
        self.init_route = init_route
        self.best_solution = None
        self.iteration_results = {}
        
        self.last_routes = None
        self.best_routes = None

    def calculate_distance_matrix(self, coords) -> np.array:
        distance_matrix = np.zeros([len(coords),len(coords)])
        for i in range(len(coords)):
            coord = coords[i]
            distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
        return distance_matrix

    def _get_route_distance(self, distance_matrix, route) -> np.float64:
        r = [0]+route+[0]
        result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
        return result

    def _get_routes_distance(self, distance_matrix: np.array, routes: list[list[int]], _print = False) -> np.float64:
        total_distance = 0
        for route in routes:
            r = [0] + route + [0]
            total_distance += np.sum([distance_matrix[r[i], r[i+1]] for i in range(len(r)-1)])
        if _print:
            print()
            print("Distance matrix shape:", distance_matrix.shape)
            print(distance_matrix)
            print(total_distance)
            print()
        return total_distance
    
    def get_neighbours(self, distance_matrix: np.array) -> list[list[int]]:
        """
            for each node, get the closest nodes nearby
        """
        n_vertices = distance_matrix.shape[0] # assumed NxN matrix
        neighbours = []
        for i in range(n_vertices):
            # this reads, for every vertex, get the neighbors sorted by the nearest distance
            index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
            # [(0, dm[0][0]), (1, dm[0][1]), (2, dm[0][2]), (3, dm[0][3])]
            sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
            # [(3, dm[0][3]), (1, dm[0][1]), (0, dm[0][0]), (2, dm[0][2])]
            neighbours.append([x[0] for x in sorted_index_dist])
            # [3, 1, 0, 2]
        # output is something like:
        # (first col may represent the node with itself (distance 0), hence the node itself)
        # [[0, 3, 1, 2], 
        #  [1, 3, 2, 0], 
        #  [2, 3, 1, 0], 
        #  [3, 2, 0, 1]] # -> if 3 and 2 is in the same place then may interchange, e.g. [2, 3, 0, 1] instead
        return neighbours
    
    def get_route_cost(self, distance_matrix, route, index_route):
        complete_r = [0] + route + [0]
        t_current = 0
        cost_current = 0
        capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
        for j in range(len(complete_r) - 1):
            capacity_current -= data.loc[complete_r[j+1], "demand"]
            if capacity_current < 0:
                if self.multi_trip:
                    t_current += distance_matrix[complete_r[j]][0]
                    t_current += distance_matrix[0][complete_r[j+1]]
                    cost_current += distance_matrix[complete_r[j]][0]
                    cost_current += distance_matrix[0][complete_r[j+1]]
                    capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
                    capacity_current -= data.loc[complete_r[j+1], "demand"]
                else:
                    t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                    cost_current += distance_matrix[complete_r[j]][complete_r[j+1]]
            if data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 0:
                # penalty datang terlalu cepat/lambat
                cost_current *= 2
            elif data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 1:
                # datang tepat waktu
                cost_current *= 0.1
        return cost_current, t_current
    
    def get_routes_cost(self, distance_matrix, routes):
        total_cost = 0
        total_time = 0
        for idx, route in enumerate(routes):
            route_costs = self.get_route_cost(distance_matrix, route, idx)
            total_cost += route_costs[0]
            total_time += route_costs[1]
        return total_cost, total_time

    def route_add(self, current_routes, c, adding_position: list[int | float]) -> list[list[int]]:
        # print("                     before route_add    :", len(current_routes))
        if adding_position[0] == -1:
            # tambah rute baru == tambah vehicle baru bjirrrr
            current_routes = current_routes + [[c]]
            # pass
        else:
            # print("route add")
            # print(current_routes)
            # print("c:", c)
            chg_r = current_routes[adding_position[0]]                            # take a route to change
            new_r = chg_r[:adding_position[1]] + [c] + chg_r[adding_position[1]:] # insert the absent node
            current_routes[adding_position[0]] = new_r                            # update the route
        # print("                     after route_add     :", len(current_routes))
        return current_routes
    
    def check_constraints(self, complete_r, index_route, distance_matrix, _print = False):
        """mengecek apakah suatu route adalah valid secara constraint: ready time, due date, dan service time
        """
        t_current = 0
        cost_current = 0
        capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
        # check whether the chosen route, when followed properly, violate any time constrains along the way or not
        # if _print:
            # print("     Checking route:", complete_r)
        for j in range(0, len(complete_r) - 1):
            # assert capacity: (route_capacity - demand) must be <= vehicle_capacity
            capacity_current -= data.loc[complete_r[j+1], "demand"]
            # if _print:
            #     print("       route           :", complete_r[:j+2])
            #     print("         demand          :", data.loc[complete_r[j+1], "demand"])
            #     print("         vehicle_capacity:", self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0)
            #     print("         capacity_current:", capacity_current)
            #     print("         route_cap_avail.:", self.vehicle_capacities[0] - capacity_current)
            #     print("         time current    :", t_current)
            #     print("         cost current    :", cost_current)
            if capacity_current < 0:
                if self.multi_trip:
                    if _print:
                        print("     OOOO PENUH. Jalur kembali dulu ke depot OOOO")
                    t_current += distance_matrix[complete_r[j]][0]
                    t_current += distance_matrix[0][complete_r[j+1]]
                    cost_current += distance_matrix[complete_r[j]][0]
                    cost_current += distance_matrix[0][complete_r[j+1]]
                    capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
                    capacity_current -= data.loc[complete_r[j+1], "demand"]
                else:
                    if _print:
                        print("     XXXX DITOLAK. Route ini tidak memiliki barang untuk memenuhi demand di node {} XXXX".format(complete_r[j+1]))
                    return False
            else:
                # increase the cost and time track
                t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                cost_current += distance_matrix[complete_r[j]][complete_r[j+1]]
            
            if t_current >= self.max_hour_per_vehicle:
                if _print:
                    print("     DITOLAK. Vehicle rute ini beroperasi lebih dari sama dengan {} jam".format(self.max_hour_per_vehicle))
                return False
            if _print:
                print("         time_passed_next:", t_current)
            # check whether the current_time progress came earlier than the ready time of the next node
            if data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 0:
                # a.k.a TOKO TUTUP
                cost_current *= 2
                if _print:
                    print("     PENALTI. Route ini datang terlalu cepat/lambat ke node {}".format(complete_r[j+1]))
            elif data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 1:
                # next node is accepted. (TOKO MASIH BUKA!!!)
                cost_current *= 0.1
                continue
        if _print:
            print("     ---- DITERIMA. Route ini dapat dipertimbangkan ----")
        return True    
    
    def ruin(self, last_routes: list[list[int]], neighbours: list[list[int]], in_absents=None, isHugeRuin=False) -> tuple[list[int], list[int]]:
        """ruin the existing routes by marking the node to be removed (absents) first using `remove_nodes` or `split_removal`
        then get the ruined routes after applying the absents towards the existing routes using `routes_summary`
        
        Returns:
        --------
            current_routes  : represents the ruined routes
            absents         : represents a list of nodes that are removed from the entire previous routes
        """
        def remove_nodes(tr: list[int], l_t: int, c: int, m: int):
            def string_removal(tr, l_t, c):
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t)
                range2 = min(i_c, len(tr)-l_t)+1
                start = np.random.randint(range1, range2)
                # print(tr[start:start+l_t])
                return tr[start:start+l_t]
            def split_removal(tr, l_t, c, m):
                additional_l = min(m, len(tr)-l_t)
                l_t_m = l_t+additional_l
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t_m)
                range2 = min(i_c, len(tr)-l_t_m)+1
                start = np.random.randint(range1, range2)
                potential_removal = tr[start:start+l_t_m]
                return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]
            if np.random.random() < 0.5:
                newly_removed = string_removal(tr, l_t, c)
            else:
                newly_removed = split_removal(tr, l_t, c, m)
                if m < (len(tr)-l_t) or np.random.random() > self.m_alpha:
                    m += 1
            return m, newly_removed
        
        def find_t(last_routes, c):
            for i in range(len(last_routes)):
                if c in last_routes[i]: return i
            return None
        
        def routes_summary(last_routes, absents):
            current_routes = []
            for r in last_routes:
                new_r = [x for x in r if x not in absents]
                if len(new_r)>0:
                    current_routes.append(new_r)
            return current_routes
        
        m = 1   # initial customers to preserve
        l_s_max = min(self.L_max, np.mean([len(x) for x in last_routes]))    # Equation (5)
        k_s_max = 4.0 * self.c_bar / (1.0 + l_s_max) - 1.0                   # Equation (6)
        k_s = int(np.random.random() * k_s_max + 1.0)                     # Equation (7)
        c_seed = int(np.random.random()*len(data))  # take random index (node)
        if in_absents is None:
            absents = []
        else:
            absents = copy.deepcopy(in_absents)
        ruined_t_indices = set([])
        # for every node in neighbours[c_seed]
        for c in neighbours[c_seed]:
            if len(ruined_t_indices) >= k_s: break
            if c not in absents and c != 0:
                t = find_t(last_routes, c)  # take the node
                if t in ruined_t_indices: continue
                if isHugeRuin and np.random.random() > 0.5:
                    newly_removed = last_routes[t]
                else:
                    l_t_max = min(l_s_max, len(last_routes[t]))                 # Equation (8)
                    l_t = int(np.random.random()*l_t_max+1.0)                   # Equation (9)
                    m, newly_removed = remove_nodes(last_routes[t], l_t, c, m)
                absents = absents + newly_removed
                ruined_t_indices.add(t)
        current_routes = routes_summary(last_routes, absents)
        return current_routes, absents

    def recreate(self, data, distance_matrix, current_routes, absents, last_length=None):
        """Algorithm 3 from SISR paper
        Managing route feasibility is done in this step for: Capacity, Time Window
        data have columns:
        # XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
        XCOORD.	YCOORD.	PICKUP	DEMAND	READY TIME	DUE DATE	EARLY_PENALTY	LATE_PENALTY	SERVICE TIME
        XCOORD          :           0
        YCOORD          :           1
        PICKUP          :           2
        DEMAND          : 2 ------> 3
        READY TIME      : 3 ------> 4
        DUE DATE        : 4 ------> 5
        EARLY_PENALTY   :           6
        LATE_PENALTY    :           7
        SERVICE_TIME    : 5 [-1] -> 8 [-1]
        """
        def _check_constraints(complete_r, index_route = 0, _print = False):
            """Mengecek apakah suatu route adalah valid secara constraint: ready time, due date, dan service time
            """
            t_current = 0
            cost_current = 0
            capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
            # check whether the chosen route, when followed properly, violate any time constrains along the way or not
            if _print:
                print("     Checking route:", complete_r)
            for j in range(0, len(complete_r) - 1):
                # assert capacity: (route_capacity - demand) must be <= vehicle_capacity
                capacity_current -= data.loc[complete_r[j+1], "demand"]
                if capacity_current < 0:
                    if self.multi_trip:
                        t_current += distance_matrix[complete_r[j]][0]
                        t_current += distance_matrix[0][complete_r[j+1]]
                        cost_current += distance_matrix[complete_r[j]][0]
                        cost_current += distance_matrix[0][complete_r[j+1]]
                        capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
                        capacity_current -= data.loc[complete_r[j+1], "demand"]
                    else:
                        return False
                else:
                    # increase the cost and time track
                    t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                    cost_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                if t_current >= self.max_hour_per_vehicle:
                    if _print:
                        print("     DITOLAK. Vehicle rute ini beroperasi lebih dari sama dengan {} jam".format(self.max_hour_per_vehicle))
                    return False
                if _print:
                    print("         time_passed_next:", t_current)
                # check whether the current_time progress came earlier than the ready time of the next node
                # a.k.a TOKO TUTUP
                if data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 0:
                    cost_current *= 2
                    if _print:
                        print("     PENALTI. Route ini datang terlalu cepat/lambat ke node {}".format(complete_r[j+1]))
                elif data.loc[complete_r[j+1], f"jam_{int(np.floor(t_current))}"] == 1:
                    # next node is accepted. increase the current time (TOKO MASIH BUKA!!!)
                    cost_current *= 0.1
                    continue
            if _print:
                print("     ---- DITERIMA. Route ini dapat dipertimbangkan ----")
            return True    
        
        def getValid_legacy(self, r, c, index_route=0):
            """Mendapatkan seluruh kemungkinan rute baru yang feasible yang dibentuk 
            dengan mencoba memasukkan node c ke dalam seluruh posisi di dalam rute r yang mungkin
            
            Returns
            ----------
            valids: list[set[int, float]]
            """
            if self.multi_trip:
                dist = self.get_route_cost(distance_matrix, r, index_route)[1]
            else:
                dist = self.get_route_distance(distance_matrix, r)
            cost = self.get_route_cost(distance_matrix, r, index_route)[0]
            complete_r = [0] + r + [0]
            valids = []
            t_current = 0
            capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
            for i in range(len(complete_r) - 1):
                # assert capacity: (route_capacity - demand) must be <= vehicle_capacity
                capacity_current -= data.loc[complete_r[i+1], "demand"]
                if capacity_current < 0:
                    if self.multi_trip:
                        # capacity kepenuhan di tengah2 di-ignore karena boleh balik depot buat refill
                        t_current += distance_matrix[complete_r[i]][0]
                        t_current += distance_matrix[0][complete_r[i+1]]
                        capacity_current = self.vehicle_capacities[index_route % self.vehicle_nums] if (index_route+1) <= self.vehicle_nums else 0
                        capacity_current -= data.loc[complete_r[i+1], "demand"]
                    else:
                        # progress so far mengakibatkan kelebihan muatan vehicle capacity. rute ditolak
                        break
                else:
                    t_current += distance_matrix[complete_r[i]][complete_r[i+1]]
                # assert time_current: < X jam
                if t_current >= self.max_hour_per_vehicle:
                    # melebihi/sama dengan X jam
                    break
                # print("         Progress lolos!")
                new_r = r[:i] + [c] + r[i:]                 # insert node c ke dalam path ke i
                if self.check_constraints([0] + new_r + [0], index_route = index_route, distance_matrix = distance_matrix, _print = False):
                    # new_distance = self.get_route_distance(distance_matrix, new_r)
                    new_distance = self.get_route_cost(distance_matrix, new_r, index_route)[1]
                    new_cost = self.get_route_cost(distance_matrix, new_r, index_route)[0]
                    # (urutan_path_dimana_node_c_dimasukkan, jarak_baru_setelah_c_masuk - jarak_baru_sebelum_c_masuk, cost_baru - cost_sebelum)
                    valids.append((i, float(new_distance - dist), float(new_cost - cost)))
            # print("     valids              :", valids)
            return valids
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir, r in enumerate(current_routes):
                # finding valid routes by finding all feasible combination of r inserted with c
                valids = getValid_legacy(self, r, c, index_route = ir)
                for v in valids:
                    # (index current_route, node_to_be_inserted, new_path_distance, new_path_cost)
                    probable_place.append((ir, v[0], v[1], v[2]))
            if len(probable_place) == 0:
                if last_length is not None and len(current_routes) >= last_length:
                    rests.append(c)
                    continue
                # ini sih nyari perkara keknya, -1 diawal nambahin absent node c jadi rute baru == vehicle baru
                adding_position = (-1, -1, 1, 1)
            else:
                # adding_position = sorted(probable_place, key=lambda x: x[2])[0]  # 2 for distance/time
                adding_position = sorted(probable_place, key=lambda x: x[3])[0]  # 3 for cost
                # print("     sorted probable_place   :", adding_position)
            # print("             added new routes    :\t {} - c: {} - adding_position: {}".format(current_routes, c, adding_position))
            current_routes = self.route_add(current_routes, c, adding_position)
            # print("     recreate() -- for {}, new_routes are: {}".format(c, current_routes))
        if last_length is not None: return current_routes, rests
        return current_routes

    def fleet_min(self, n, data, distance_matrix, neighbours, routes, verbose_step):
        # overall algorithm. used ruin and recreate
        absents = []
        absent_c = np.zeros(distance_matrix.shape[0])
        best_routes = copy.deepcopy(routes)
        last_routes = copy.deepcopy(routes)
        for i in range(n):
            current_routes, new_absents = self.ruin(last_routes, neighbours, absents)
            current_routes, new_absents = self.recreate(data, distance_matrix, current_routes, new_absents, last_length=len(last_routes))
            if len(new_absents) == 0:
                # line 9 of algorithm 4 in the paper
                best_routes = copy.deepcopy(current_routes)
                # t is a numpy array. so absent_c[t] means getting all the associated absent counter of each node in the path t
                # then sort the list by the lowest sum of absents to the highest
                sumabs = sorted([ (t, np.sum(absent_c[t])) for t in best_routes ], key=lambda x: x[1])
                # take the best absents by the lowest sum of absents
                absents = sumabs[0][0]
                last_routes = [x[0] for x in sumabs[1:]]
                current_routes = copy.deepcopy(last_routes)
            elif len(new_absents) < len(absents) or np.sum(absent_c[new_absents]) < np.sum(absent_c[absents]):
                # update the best solution. line 7-8 of algorithm 4 in the paper
                last_routes = copy.deepcopy(current_routes)
                absents = copy.deepcopy(new_absents)
            if len(new_absents)>0:
                # absent counter. line 12 of algorithm 4 in the paper
                tmp = np.zeros(absent_c.shape)
                tmp[new_absents]=1
                absent_c = absent_c+tmp
            if verbose_step is not None and (i+1)%verbose_step==0:
                print("fleet_min", i+1, np.round((i+1)/n*100,4), "%:", len(best_routes))
        if verbose_step is not None and n%verbose_step!=0: print(i+1, "100.0 %:", len(best_routes))
        return best_routes
    
    def plot_routes(self, data, routes, title=None, output_folder="frames", iteration=0, ruin=False):
        coords = self.data.loc[:, ["X_coord", "Y_coord"]].to_numpy()
        routes = copy.deepcopy(routes)
        # add node 0 for beginning and end
        for idx_route in range(len(routes)):
            routes[idx_route] = [0] + routes[idx_route] + [0]
        # Create a color cycle
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        for route, color in zip(routes, colors):
            plt.plot(coords[0, 0], coords[0, 1], 'C2o', markersize=18)
            plt.annotate("Depot", (coords[0, 0], coords[0, 1]))
            plt.plot(coords[1:, 0], coords[1:, 1], 'C1o')
            for i in range(len(coords)): 
                plt.annotate(
                    i, (coords[i, 0], coords[i, 1]),
                    # arrowprops = dict(arrowstyle="->", mutation_scale=20),
                )
            for i in range(len(route)-1):
                plt.plot(
                    [coords[route[i], 0], coords[route[i+1], 0]], 
                    [coords[route[i], 1], coords[route[i+1], 1]],
                    color=color, linestyle='-', 
                    label="Route: {}".format(route) if i == 0 else None # to prevent multiple legend for the same route/color.
                )
        if title is not None:
            ax.set_title("SISR: Ruin and Recreate Simulation\n" + title)
        else:
            ax.set_title("SISR: Ruin and Recreate Simulation")
        legend = plt.legend(loc="upper right", edgecolor="black")
        legend.get_frame().set_alpha(None)
        legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        ax.legend(bbox_to_anchor=(1.1, 1.0))
        plt.tight_layout()
        
        # Save frame
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if ruin and title is not None:
            plt.savefig(os.path.join(output_folder, f"frame_{f'{iteration}'.zfill(6)}-1-ruin.png"))
        elif title is not None:
            plt.savefig(os.path.join(output_folder, f"frame_{f'{iteration}'.zfill(6)}-2-recreate.png"))
        elif title is None:
            # initial plot
            plt.savefig(os.path.join(output_folder, f"frame_{f'{iteration}'.zfill(6)}.png"))
        plt.close(fig)

    def greedy_initial_solution(self, distance_matrix):
        # node customer ditelusuri satu-satu dari 1-N
        best_routes = []
        route = []
        route_index = 0
        current_distance = 0
        current_capacity = self.vehicle_capacities[route_index]
        i = 0
        while i < (len(self.data)-1):
            current_distance += distance_matrix[i][i+1]
            current_capacity -= self.data.loc[i, "demand"]
            if current_capacity >= 0:
                route.append(i+1)
                print(route)
                if (i+1) == len(self.data.iloc[1:]):
                    best_routes.append(route)
                    break
                i += 1
            elif current_capacity < 0:
                print("     NEW ROUTE   :", route)
                best_routes.append(route)
                route = []
                current_distance = 0
                route_index += 1
                current_capacity = self.vehicle_capacities[route_index]
        return best_routes
    
    def run(self):
        print("Hyperparameter Settings:")
        alpha_T = (self.final_T/self.init_T)**(1.0/self.n_iter)    # Equation (2) | so alpha_T is c in the paper
        print("init_T               :", self.init_T)
        print("final_T              :", self.final_T)
        print("n_iter               :", self.n_iter)
        print("alpha_T              :", alpha_T)
        print("n_iter_fleet         :", self.n_iter_fleet)
        print("vehicle_capacities   :", self.vehicle_capacities)
        print("vehicle_nums         :", self.vehicle_nums)
        print("max_hour_per_vehicle :", self.max_hour_per_vehicle)
        
        coords = self.data.loc[:, ["X_coord", "Y_coord"]].to_numpy()
        print("Data Type")
        print(coords.shape, self.data.shape)
        distance_matrix = np.zeros([len(coords), len(coords)])
        for i in range(len(coords)):
            coord = coords[i]
            distance_matrix[i] = np.sum((coord-coords)**2, axis=1)**0.5 # euclidean distance
        
        if self.init_route is not None:
            best_routes = copy.deepcopy(self.init_route)
        else:
            print("No Initial Route")
            print("Generate Several Best Routes")
            print("Initial routes: There are {} numbers of routes".format(len(range(1, len(self.data)))))
            # best_routes = [[i] for i in range(1, len(data))]
            best_routes = [[i] for i in range(1, len(data))]
            # if not self.multi_trip:
            #     best_routes = self.greedy_initial_solution(distance_matrix=distance_matrix)
            # else:
            #     best_routes = np.array_split([i for i in range(1, len(data))], self.vehicle_nums)
            #     best_routes = [arr.tolist() for arr in best_routes]
            print("Best Routes:\n", best_routes)
            # # check whether initial routes are valid
            best_routes_valid = True
            for ir, r in enumerate(best_routes):
                if not self.check_constraints([0] + r + [0], index_route = ir, distance_matrix = distance_matrix, _print = False):
                    best_routes_valid = False
                    print("Initial routes are not valid")
                    print(r)
                    # return None, None, None
                    pass
        print()
        print("Get First Routes Distance")
        # best_distance = self.get_routes_distance(distance_matrix, best_routes, _print = False)
        best_distance = self.get_routes_cost(distance_matrix, best_routes)[1]
        best_cost = self.get_routes_cost(distance_matrix, best_routes)[0]
        
        last_routes = copy.deepcopy(best_routes)
        # last_distance = self.get_routes_distance(distance_matrix, best_routes)
        last_distance = self.get_routes_cost(distance_matrix, best_routes)[1]
        last_cost = self.get_routes_cost(distance_matrix, best_routes)[0]
        
        neighbours = self.get_neighbours(distance_matrix)
        print(len(best_routes), best_distance)
        # plot_routes(data, best_routes)
        
        if (self.n_iter_fleet is not None) and (self.n_iter_fleet > 0):
            last_routes = self.fleet_min(self.n_iter_fleet, self.data, distance_matrix, neighbours, best_routes, self.verbose_step)
            # last_distance = self.get_routes_distance(distance_matrix, last_routes)
            last_distance = self.get_routes_cost(distance_matrix, last_routes)[1]
            last_cost = self.get_routes_cost(distance_matrix, last_routes)[0]
            if last_cost < best_cost:
                best_distance = last_distance
                best_routes = last_routes
                best_cost = last_cost
            print(len(best_routes), best_distance)
        
        temperature = self.init_T
        # the algorithm
        for i_iter in range(self.n_iter):
            if self.obj_n_routes is not None and len(last_routes) > self.obj_n_routes and (i_iter+1) % self.fleet_gap == 0:
                # jika jumlah route lebih banyak daripada batas rute yang ditentukan serta sisa bagi iterasi dengan fleet_gap == 0
                current_routes = self.fleet_min(self.n_iter_fleet, self.data, distance_matrix, neighbours, last_routes, self.verbose_step)
            else:
                current_routes, absents = self.ruin(last_routes, neighbours)
                ruined_routes = copy.deepcopy(current_routes)
                current_routes = self.recreate(self.data, distance_matrix, current_routes, absents)
            
            # current_distance = self.get_routes_distance(distance_matrix, current_routes)
            current_distance = self.get_routes_cost(distance_matrix, current_routes)[1]
            current_cost = self.get_routes_cost(distance_matrix, current_routes)[0]
            # Equation (4) for Acceptance Criterion
            if len(current_routes) < len(best_routes) or \
                ( current_cost < (last_cost - temperature*np.log(np.random.random())) and \
                len(current_routes) <= len(best_routes) ):
                # Check whether num. of current routes < num of best routes OR current distance < best distance
                # Basically update the best routes & distance IFF the total distance is shorter OR the number of routes is lesser
                # print("Check current_cost < best_cost")
                if len(current_routes) < len(best_routes) or current_cost < best_cost:
                    best_distance = current_distance
                    best_routes = current_routes
                    best_cost = current_cost
                    if self.test_obj is not None and best_cost < self.test_obj:
                        break
                last_distance = current_distance
                last_routes = current_routes
                last_cost = current_cost
            # Equation (1)
            temperature *= alpha_T # Basically temperature = previous_temperature X c
            
            if self.verbose_step is not None and (i_iter+1) % self.verbose_step == 0:
                print("="*100)
                print("Iteration                : {}/{} ({}%)".format(i_iter+1, self.n_iter, np.round((i_iter+1)/self.n_iter*100, 4)))
                print("Best Routes              :")
                for idx, _route in enumerate(best_routes):
                    print("{}\t(length={})\t(time_distance={:.4f})\t(cost={:.4f}): {}".format(
                        idx, len(_route), 
                        self.get_route_cost(distance_matrix, _route, idx)[1], 
                        self.get_route_cost(distance_matrix, _route, idx)[0],
                        _route
                    ))
                    # if self.soft_time_window:
                    #     route_load = []
                    #     route_capacity = self.vehicle_capacities[idx]
                    #     for _node in _route[:-1]:
                    #         route_capacity = route_capacity - int(self.data.loc[_node, "demand"])
                    #         route_load.append(route_capacity)
                    #     print(f"\t(capacity={route_load})")
                print("Best Routes length       : {}".format(len(best_routes)))
                print("Best Total time          : {}".format(best_distance))
                print("Best Cost                : {}".format(best_cost))
                print()
                
                if (i_iter+1) % (self.verbose_step*10) == 0:
                    self.plot_routes(self.data, ruined_routes, title="Iteration: {} (Ruin)".format(i_iter+1), iteration=i_iter+1, ruin=True)
                    self.plot_routes(self.data, best_routes, title="Iteration: {} (Recreate)".format(i_iter+1), iteration=i_iter+1)
        
        import imageio
        def create_animation_from_frames(output_folder='frames', animation_path='routes_animation.gif', fps=2):
            frames = []
            filenames = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
            for filename in filenames:
                frames.append(imageio.v2.imread(os.path.join(output_folder, filename)))
            imageio.mimsave(animation_path, frames, fps=fps)
            
        # Create animation
        create_animation_from_frames()
        
        if self.verbose_step is not None and self.n_iter % self.verbose_step != 0: 
            print(i_iter+1, "100.0 %:", best_distance)
        return best_distance, best_cost, best_routes


if __name__ == "__main__":
    import time
    import numpy as np
    import pandas as pd
    
    def parse_hcvrp_question(pickle_file_path):
        import pickle
        with open(pickle_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        instance_num = 0
        depot_coordinates = loaded_data[instance_num][0]
        customer_coordinates = loaded_data[instance_num][1]
        customer_demands = loaded_data[instance_num][2]
        vehicle_capacities = loaded_data[0][3]
        # vehicle_capacities = [80, 100, 60]
        time_open = np.asarray([[1 for _ in range(24)]] + loaded_data[0][4], dtype=np.int8)
        data = {
            "X_coord": [depot_coordinates[0]] + [coord[0] for coord in customer_coordinates],
            "Y_coord": [depot_coordinates[1]] + [coord[1] for coord in customer_coordinates],
            "demand": [0] + customer_demands,
        }
        data = pd.DataFrame(data)
        time_open = pd.DataFrame(time_open, columns=[f"jam_{i}" for i in range(24)])
        data = pd.concat([data, time_open], axis=1)
        return vehicle_capacities, data
    
    def parse_vrp_question(file_path):
        data = []
        with open(file_path, 'r') as f:
            i = 0
            for line in f:
                sline = line.strip()
                if len(sline)==0: continue
                if i==3: vehicle_capacity = int(sline.split()[1])
                if i>=6:
                    data.append([int(i) for i in sline.split()][1:])
                i+=1
        return vehicle_capacity, np.array(data)
    
    # vehicle_capacities, data = parse_vrp_question("data/sample_100.txt")
    # print(data)
    # vehicle_nums = 3
    # vehicle_capacities = [vehicle_capacities for _ in range(vehicle_nums)]
    
    vehicle_capacities, data = parse_hcvrp_question("data/hcvrp/hcvrp_v3_40_seed24610_tw.pkl")
    print("Vehicle Capacity :", vehicle_capacities)
    print("data.shape       :", data.shape)
    # print(data.iloc[:, :])
    print("------------------")

    start_time = time.time()
    np.random.seed(0)
    sisr = SISR(
        data,
        vehicle_capacities = vehicle_capacities,
        n_iter = 1_000,
        init_T = 100.0,
        final_T = 1.0,
        n_iter_fleet = 10,
        fleet_gap = 2,
        obj_n_routes = 3,
        init_route = None,
        verbose_step = 100,
        time_window = True,
        soft_time_window = True,
        max_hour_per_vehicle = 8
    )
    best_distance, best_cost, best_routes = sisr.run()
    time_cost = time.time() - start_time
    print("running_time     :", time_cost)
    print("best_cost        :", best_cost)
    print("time_distance    :", best_distance)
    print("best_routes      :", best_routes)
    

