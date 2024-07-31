import copy
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_distance_matrix(coords):
    distance_matrix = np.zeros([len(coords),len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2,axis=1)**0.5
    return distance_matrix

def get_route_distance(distance_matrix, route):
    r = [0]+route+[0]
    result = np.sum([distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1)])
    return result

def get_routes_distance(
    distance_matrix: np.array, 
    routes: list[list[int]], 
    _print = False
) -> np.float64:
    if _print:
        print()
        print("Distance matrix shape:", distance_matrix.shape)
        print(distance_matrix)
    total_distance = 0
    for route in routes:
        r = [0] + route + [0]
        total_distance += np.sum([distance_matrix[r[i], r[i+1]] for i in range(len(r)-1)])
    if _print:
        print(total_distance)
        print()
    return total_distance

def get_neighbours(distance_matrix: np.array) -> list:
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
    # output is something like
    # first col may represent the node with itself (distance 0), hence the node itself
    # [[0, 3, 1, 2], 
    #  [1, 3, 2, 0], 
    #  [2, 3, 1, 0], 
    #  [3, 2, 0, 1]] # -> if 3 and 2 is in the same place then may interchange, e.g. [2, 3, 0, 1] instead
    return neighbours

def route_add(current_routes, c, adding_position: list[int | float]):
    if adding_position[0] == -1:
        current_routes = current_routes + [[c]]
    else:
        # print("route add")
        # print(current_routes)
        # print("c:", c)
        chg_r = current_routes[adding_position[0]]                            # take a route to change
        new_r = chg_r[:adding_position[1]] + [c] + chg_r[adding_position[1]:] # insert the absent node
        current_routes[adding_position[0]] = new_r                            # update the route
    return current_routes

def sisr_cvrp(
    data: np.array,
    vehicle_capcity: int,
    n_iter: int,
    n_iter_fleet: int = None,
    fleet_gap: int = None,
    init_T: float = 100.0,
    final_T: float = 1.0,
    c_bar: float = 10.0,
    L_max: float = 10.0,
    m_alpha: float = 0.01,
    obj_n_routes = None,
    init_route = None,
    verbose_step = None,
    test_obj = None,
    time_window = False,
    soft_time_window = False
) -> tuple[int | float, list[list[int]]]:
    """
    Returns
    ----------
    The best distance & the best routes
    """
    
    def ruin(last_routes: list[list[int]], neighbours: list[list[int]], in_absents=None, isHugeRuin=False) -> tuple[list[int], list[int]]:
        """
            ruin the existing routes by marking the node to be removed (absents) first using `remove_nodes` or `split_removal`
            then get the ruined routes after applying the absents towards the existing routes using `routes_summary`
        Returns:
            current_routes  : represents the ruined routes
            absents         : represents a list of nodes that are removed from the entire previous routes
        """
        def remove_nodes(tr: list[int], l_t: int, c: int, m: int):
            def string_removal(tr, l_t, c):
                # print("string removal")
                # print(tr)
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t)
                range2 = min(i_c, len(tr)-l_t)+1
                start = np.random.randint(range1, range2)
                # print(tr[start:start+l_t])
                return tr[start:start+l_t]
            def split_removal(tr, l_t, c, m):
                # print("split removal")
                # print(tr)
                additional_l = min(m, len(tr)-l_t)
                l_t_m = l_t+additional_l
                i_c = tr.index(c)
                range1 = max(0, i_c+1-l_t_m)
                range2 = min(i_c, len(tr)-l_t_m)+1
                start = np.random.randint(range1, range2)
                potential_removal = tr[start:start+l_t_m]
                # print(potential_removal)
                return [potential_removal[i] for i in np.random.choice(l_t_m, l_t, replace=False)]
            if np.random.random() < 0.5:
                newly_removed = string_removal(tr, l_t, c)
            else:
                newly_removed = split_removal(tr, l_t, c, m)
                if m < (len(tr)-l_t) or np.random.random() > m_alpha:
                    m += 1
            return m, newly_removed
        
        def find_t(last_routes, c):
            for i in range(len(last_routes)):
                if c in last_routes[i]: return i
            return None
        
        def routes_summary(last_routes, absents):
            # print("routes summary")
            current_routes = []
            for r in last_routes:
                # print(r)
                new_r = [x for x in r if x not in absents]
                if len(new_r)>0:
                    current_routes.append(new_r)
            # print("absents")
            # print(absents)
            # print("routes summary - current routes")
            # for r in current_routes:
                # print(r)
            return current_routes
        
        m = 1   # initial customers to preserve
        l_s_max = min(L_max, np.mean([len(x) for x in last_routes]))    # Equation (5)
        k_s_max = 4.0 * c_bar / (1.0 + l_s_max) - 1.0                   # Equation (6)
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
    
    def recreate_single_demand(data, dist_m, current_routes, absents, last_length=None):
        """Algorithm 3 from SISR paper.
        Managing route feasibility is done in this step for: Capacity
        """
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:   # for c \in A
            probable_place = [] # P <- NULL
            # current_routes = [[1, 2, 10, 5], [4, 3, 11, 8], ...]
            for ir, r in enumerate(current_routes): # for t \in T
                # r = [1, 2, 10, 5]
                # assert capacity --> checking constraint
                # reads: node x and column 2
                if (np.sum([data[x][2] for x in r]) + data[c][2]) > vehicle_capcity: 
                    # adding the node for the route candidate will increase the vehicle capacity's limit, hence skip
                    # finding valid routes by finding all feasible combination of r inserted with c
                    continue
                # MUNGKIN BISA TAMBAH LAGI PENGECEKAN CONSTRAINT hard time window / soft time window dari sini
                complete_r = [0] + r + [0]
                for iri in range(len(r)+1): # for P_t in t (?)
                    # len(r)+1 karena dist_m itu ada N customers + 1 depot node
                    # (0, 0, new_cost_of_node_distances)
                    # (0, 1, new_cost_of_node_distances)
                    # (0, 2, new_cost_of_node_distances)
                    # (0, 3, new_cost_of_node_distances)
                    probable_place.append((ir, iri, dist_m[iri, c] + dist_m[c, iri+1] - dist_m[iri, iri+1]))
            if len(probable_place) == 0:
                if last_length is not None and len(current_routes) >= last_length:
                    rests.append(c)
                    continue
                adding_position = (-1, -1, 1)
            else:
                # sort by the fewest cost AND take the update with the lowest cost
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
                # adding_position: (index_route, index_node_in_route, cost)
#             probable_place.append((-1,-1,2*distance_matrix[0,c]))
#             adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes
    
    def _recreate_single_demand_time_window(data, distance_matrix, current_routes, absents, last_length=None):
        """Algoritmh 3 from SISR paper
        Managing route feasibility is done in this step for: Capacity, Time Window
        data have columns:
        XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME
        """
        def checkValid_legacy(complete_r):
            """mengecek apakah suatu route adalah valid secara constraint: ready time, due date, dan service time
            """
            t_current = 0
            # check whether the chosen route, when followed properly, violate any time constrains along the way or not
            for j in range(0, len(complete_r)-1):
                t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                # take which one is higher:
                # ready time of the next node OR the current time
                t_current = max(data[complete_r[j+1]][3], t_current)
                # check whether the current_time progress exceed the due date of the next node
                # a.k.a TOKO SUDAH TUTUP
                if t_current <= data[complete_r[j+1]][4]:
                    # next node is accepted. increase the current time (TOKO MASIH BUKA!!!)
                    # current_time = current time + service time
                    t_current += data[complete_r[j+1]][-1]
                else:
                    # next node is not valid because violated the due date.
                    # just stop searching.
                    # self-note: MAYYYBE kalo mau kasih penalty atau soft-time window masukin di sini
                    return False
            return True
        
        def getValid_legacy(r, c):
            """Mendapatkan seluruh kemungkinan rute baru yang feasible yang dibentuk 
            dengan mencoba memasukkan node c ke dalam seluruh posisi di dalam rute r yang mungkin
            
            Returns:
            ----------
            valids: list[set[int, float]]
            """
            dist = get_route_distance(distance_matrix, r)
            complete_r = [0] + r + [0]
            valids = []
            tmp_t = 0
            for i in range(len(complete_r)-1):
                tmp_t = max(tmp_t, data[complete_r[i]][3])
                tmp_t += data[complete_r[i]][-1]
                if tmp_t + distance_matrix[complete_r[i]][c] > data[c][4]: 
                    break
                new_r = r[:i] + [c] + r[i:]
                if checkValid_legacy([0] + new_r + [0]):
                    new_d = get_route_distance(distance_matrix, new_r)
                    valids.append((i, new_d - dist))
                tmp_t += distance_matrix[complete_r[i]][complete_r[i+1]]
            return valids
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir, r in enumerate(current_routes):
                # assert capacity
                if (np.sum([data[x][2] for x in r]) + data[c][2]) > vehicle_capcity: 
                    continue
                # finding valid routes by finding all feasible combination of r inserted with c
                valids = getValid_legacy(r, c)
                for v in valids:
                    # (index current_route, node_to_be_inserted, new_path_cost)
                    probable_place.append((ir, v[0], v[1]))
            if len(probable_place) == 0:
                if last_length is not None and len(current_routes) >= last_length:
                    rests.append(c)
                    continue
                adding_position = (-1, -1, 1)
            else:
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
#             probable_place.append((-1,-1,2*distance_matrix[0,c]))
#             adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes
    
    def recreate_single_pickup_demand_hard_time_window(data, distance_matrix, current_routes, absents, last_length=None):
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
        def check_constraints(complete_r, _print = False):
            """mengecek apakah suatu route adalah valid secara constraint: ready time, due date, dan service time
            """
            t_current = 0
            route_capacity = vehicle_capcity
            # check whether the chosen route, when followed properly, violate any time constrains along the way or not
            if _print:
                print("     Checking route:", complete_r)
            for j in range(0, len(complete_r) - 1):
                # assert capacity: (route_capacity + pickup - demand) must be <= vehicle_capacity
                route_capacity = route_capacity + data[complete_r[j+1]][2] - data[complete_r[j+1]][3]
                if _print:
                    print("       route           :", complete_r[:j+2])
                    print("         pickup          :", data[complete_r[j+1]][2])
                    print("         demand          :", data[complete_r[j+1]][3])
                    print("         route_capacity  :", route_capacity)
                    print("         route_cap_avail.:", vehicle_capcity - route_capacity)
                if route_capacity > vehicle_capcity:
                    # route_capacity surpassed vehicle capacity
                    if _print:
                        print("     DITOLAK. Route ini akan keberatan beban di node {}".format(complete_r[j+1]))
                    return False
                
                # assert time window
                t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                if _print:
                    print("         time_passed     :", t_current)
                    print("         ready time      :", data[complete_r[j+1]][4])
                    print("         due time        :", data[complete_r[j+1]][5])
                # take which one is higher:  ready time of the next node OR the current time
                # self-note: SEPERTINYA diizinkan datang terlalu cepat
                # self-note: MAYYYBE kalo mau kasih penalty kecepetan di sini
                t_current = max(data[complete_r[j+1]][4], t_current)
                # check whether the current_time progress exceed the due date of the next node
                # a.k.a TOKO SUDAH TUTUP
                if t_current <= data[complete_r[j+1]][5]:
                    # next node is accepted. increase the current time (TOKO MASIH BUKA!!!)
                    # current time = current time + service time
                    t_current += data[complete_r[j+1]][-1]
                    if _print:
                        print("         service time    :", data[complete_r[j+1]][-1])
                else:
                    # next node is not valid because violating the due date. stop checking the path
                    # self-note: MAYYYBE kalo mau kasih penalty atau soft-time window masukin di sini
                    if _print:
                        print("     DITOLAK. Route ini akan terlambat di node {}".format(complete_r[j+1]))
                    return False
            if _print:
                print("     DITERIMA. Route ini dapat dipertimbangkan")
            return True
        
        def getValid_legacy(r, c):
            """Mendapatkan seluruh kemungkinan rute baru yang feasible yang dibentuk 
            dengan mencoba memasukkan node c ke dalam seluruh posisi di dalam rute r yang mungkin
            
            Returns:
            ----------
            valids: list[set[int, float]]
            """
            dist = get_route_distance(distance_matrix, r)
            complete_r = [0] + r + [0]
            valids = []
            tmp_t = 0
            tmp_capacity = vehicle_capcity
            for i in range(len(complete_r) - 1):
                # assert capacity: (route_capacity + pickup - demand) must be <= vehicle_capacity
                tmp_capacity = tmp_capacity + data[complete_r[i]][2] - data[complete_r[i]][3]
                if tmp_capacity > vehicle_capcity:
                    # progress so far mengakibatkan kelebihan muatan vehicle capacity
                    break
                
                # assert time window: dateng kecepetan boleh, dateng terlambat route ditolak
                tmp_t = max(tmp_t, data[complete_r[i]][4])  # datang boleh kecepetan
                tmp_t += data[complete_r[i]][-1]            # ditambah waktu pelayanan
                if tmp_t + distance_matrix[complete_r[i]][c] > data[c][5]: 
                    # progress so far mengakibatkan terlambat
                    break
                # SELF-NOTE: jika sampai tahap ini, berarti bahkan rute sebelum di-insert pun bermasalahn
                
                new_r = r[:i] + [c] + r[i:]                 # insert node c ke dalam path ke i
                if check_constraints([0] + new_r + [0], _print = False):
                    new_d = get_route_distance(distance_matrix, new_r)
                    # (urutan_path_dimana_node_c_dimasukkan, jarak_baru_setelah_c_masuk - jarak_baru_sebelum_c_masuk)
                    valids.append((i, new_d - dist))
                tmp_t += distance_matrix[complete_r[i]][complete_r[i+1]]
            return valids
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir, r in enumerate(current_routes):
                # SELF-NOTE: pengecekan kapasitas DIPINDAHKAN
                # INI PERLU DIBENERIN KARENA KALO KESELURUHAN PATH DIITUNG GINI, BISA AJA 
                # JALUR AWAL NGUTANG DARI JALUR AKHIR
                # PAKE CHECKING NODE ONE-BY-ONE DI PATH-NYA daripada checking dengan np.sum()
                # ATO MUNGKIN MASUKIN AJA KE getValid_legacy (tapi keknya bakalan aneh logic-wise....)
                # if (np.sum([data[x][3] for x in r]) + data[c][3] - np.sum([data[x][2] for x in r])) > vehicle_capcity: 
                #     continue
                # finding valid routes by finding all feasible combination of r inserted with c
                valids = getValid_legacy(r, c)
                for v in valids:
                    # (index current_route, node_to_be_inserted, new_path_cost)
                    probable_place.append((ir, v[0], v[1]))
            if len(probable_place) == 0:
                if last_length is not None and len(current_routes) >= last_length:
                    rests.append(c)
                    continue
                adding_position = (-1, -1, 1)
            else:
                adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
#             probable_place.append((-1,-1,2*distance_matrix[0,c]))
#             adding_position = sorted(probable_place, key=lambda x: x[-1])[0]
            current_routes = route_add(current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes

    def recreate_single_pickup_demand_soft_time_window(data, distance_matrix, current_routes, absents, last_length=None):
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
        def check_constraints(complete_r, _print = False):
            """mengecek apakah suatu route adalah valid secara constraint: ready time, due date, dan service time
            """
            t_current = 0
            cost_current = 0
            capacity_current = vehicle_capcity
            pickup_current = 0
            demand_current = vehicle_capcity
            # check whether the chosen route, when followed properly, violate any time constrains along the way or not
            if _print:
                print("     Checking route:", complete_r)
            for j in range(0, len(complete_r) - 1):
                # assert capacity: (route_capacity + pickup - demand) must be <= vehicle_capacity
                # capacity_current = capacity_current + data[complete_r[j+1]][2] - data[complete_r[j+1]][3]
                pickup_current += data[complete_r[j+1]][2]
                demand_current -= data[complete_r[j+1]][3]
                capacity_current += data[complete_r[j+1]][2] - data[complete_r[j+1]][3]
                if _print:
                    print("       route           :", complete_r[:j+2])
                    print("         pickup          :", data[complete_r[j+1]][2])
                    print("         demand          :", data[complete_r[j+1]][3])
                    print("         capacity_current:", capacity_current)
                    print("         pickup_current  :", pickup_current)
                    print("         demand_current  :", demand_current)
                    print("         route_cap_avail.:", vehicle_capcity - capacity_current)
                    print("         time current    :", t_current)
                    print("         cost current    :", cost_current)
                if demand_current < 0:
                    if _print:
                        print("     DITOLAK. Route ini tidak memiliki barang untuk memenuhi demand di node {}".format(complete_r[j+1]))
                    return False
                if pickup_current > 200:
                    if _print:
                        print("     DITOLAK. Route ini tidak dapat menampung pickup di node {}".format(complete_r[j+1]))
                    return False
                if capacity_current > vehicle_capcity:
                # if (vehicle_capcity - route_capacity) < 0:
                    # route_capacity surpassed vehicle capacity
                    if _print:
                        print("     DITOLAK. Route ini akan keberatan beban di node {}".format(complete_r[j+1]))
                    return False
                
                # asumsi cost == distance
                t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                cost_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                if _print:
                    print("         time_passed     :", t_current)
                    print("         ready time      :", data[complete_r[j+1]][4])
                    print("         due time        :", data[complete_r[j+1]][5])
                    print("         service time    :", data[complete_r[j+1]][-1])
                # check whether the current_time progress came earlier than the ready time of the next node
                # a.k.a TOKO MASIH TUTUP
                if t_current < data[complete_r[j+1]][4]:
                    cost_current += data[complete_r[j+1]][6]
                    if _print:
                        print("     PENALTI. Route ini datang terlalu cepat ke node {}".format(complete_r[j+1]))
                # check whether the current_time progress exceed the due date of the next node
                # a.k.a TOKO SUDAH TUTUP
                if t_current <= data[complete_r[j+1]][5]:
                    # next node is accepted. increase the current time (TOKO MASIH BUKA!!!)
                    t_current += data[complete_r[j+1]][-1]
                    cost_current += data[complete_r[j+1]][-1]
                else:
                    if _print:
                        print("     PENALTI. Route ini datang terlambat ke node {}".format(complete_r[j+1]))
                    t_current += data[complete_r[j+1]][-1]
                    cost_current += data[complete_r[j+1]][-1] + data[complete_r[j+1]][7]
            if _print:
                print("     ---- DITERIMA. Route ini dapat dipertimbangkan ----")
            return True
        
        def get_cost(distance_matrix, route):
            complete_r = [0] + route + [0]
            t_current = 0
            cost_current = 0
            for j in range(len(complete_r) - 1):
                t_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                cost_current += distance_matrix[complete_r[j]][complete_r[j+1]]
                if t_current < data[complete_r[j+1]][4]:
                    # penalty datang terlalu cepat
                    cost_current += data[complete_r[j+1]][6]
                if t_current <= data[complete_r[j+1]][5]:
                    t_current += data[complete_r[j+1]][-1]
                    cost_current += data[complete_r[j+1]][-1]
                else:
                    # penalty datang terlalu lambat
                    t_current += data[complete_r[j+1]][-1]
                    cost_current += data[complete_r[j+1]][-1] + data[complete_r[j+1]][7]
            return cost_current
        
        def getValid_legacy(r, c):
            """Mendapatkan seluruh kemungkinan rute baru yang feasible yang dibentuk 
            dengan mencoba memasukkan node c ke dalam seluruh posisi di dalam rute r yang mungkin
            
            Returns:
            ----------
            valids: list[set[int, float]]
            """
            dist = get_route_distance(distance_matrix, r)
            cost = get_cost(distance_matrix, r)
            complete_r = [0] + r + [0]
            valids = []
            tmp_t = 0
            tmp_capacity = vehicle_capcity
            for i in range(len(complete_r) - 1):
                # assert capacity: (route_capacity + pickup - demand) must be <= vehicle_capacity
                tmp_capacity = tmp_capacity + data[complete_r[i+1]][2] - data[complete_r[i+1]][3]
                if tmp_capacity > vehicle_capcity:
                    # progress so far mengakibatkan kelebihan muatan vehicle capacity
                    break
                
                new_r = r[:i] + [c] + r[i:]                 # insert node c ke dalam path ke i
                if check_constraints([0] + new_r + [0], _print = False):
                    new_distance = get_route_distance(distance_matrix, new_r)
                    new_cost = get_cost(distance_matrix, new_r)
                    # (urutan_path_dimana_node_c_dimasukkan, jarak_baru_setelah_c_masuk - jarak_baru_sebelum_c_masuk, cost_baru - cost_sebelum)
                    valids.append((i, new_distance - dist, new_cost - cost))
            return valids
        
        # 1. Sort the absent nodes with some methods, here only use random as a placeholder
        absents = [absents[i] for i in np.random.choice(len(absents), len(absents), replace=False)]
        if last_length is not None: rests = []
        for c in absents:
            probable_place = []
            for ir, r in enumerate(current_routes):
                # finding valid routes by finding all feasible combination of r inserted with c
                valids = getValid_legacy(r, c)
                for v in valids:
                    # (index current_route, node_to_be_inserted, new_path_distance, new_path_cost)
                    probable_place.append((ir, v[0], v[1], v[2]))
            if len(probable_place) == 0:
                if last_length is not None and len(current_routes) >= last_length:
                    rests.append(c)
                    continue
                adding_position = (-1, -1, 1, 1)
            else:
                # adding_position = sorted(probable_place, key=lambda x: x[2])[0]  # 2 for distance/time
                adding_position = sorted(probable_place, key=lambda x: x[3])[0]  # 3 for cost
            current_routes = route_add(current_routes, c, adding_position)
        if last_length is not None: return current_routes, rests
        return current_routes
    
    def fleet_min(n, data, distance_matrix, neighbours, routes, verbose_step):
        absents = []
        absent_c = np.zeros(distance_matrix.shape[0])
        best_routes = copy.deepcopy(routes)
        last_routes = copy.deepcopy(routes)
        for i in range(n):
            current_routes, new_absents = ruin(last_routes, neighbours, absents)
            current_routes, new_absents = recreate(data, distance_matrix, current_routes, new_absents, last_length=len(last_routes))
            if len(new_absents)==0:
                best_routes = copy.deepcopy(current_routes)
                sumabs = sorted([(t, np.sum(absent_c[t])) for t in best_routes], key=lambda x: x[1])
                absents = sumabs[0][0]
                last_routes = [x[0] for x in sumabs[1:]]
                current_routes = copy.deepcopy(last_routes)
            elif len(new_absents)<len(absents) or np.sum(absent_c[new_absents])<np.sum(absent_c[absents]):
                last_routes = copy.deepcopy(current_routes)
                absents = copy.deepcopy(new_absents)
            if len(new_absents)>0:
                tmp = np.zeros(absent_c.shape)
                tmp[new_absents]=1
                absent_c = absent_c+tmp
            if verbose_step is not None and (i+1)%verbose_step==0:
                print("fleet_min", i+1, np.round((i+1)/n*100,4), "%:", len(best_routes))
        if verbose_step is not None and n%verbose_step!=0: print(i+1, "100.0 %:", len(best_routes))
        return best_routes
    
    if time_window:
        if soft_time_window:
            recreate = recreate_single_pickup_demand_soft_time_window
            # recreate = recreate_single_pickup_demand_hard_time_window
        else:
            recreate = recreate_single_pickup_demand_hard_time_window
    else:
        recreate = recreate_single_demand
    
    print("Hyperparameter Settings:")
    alpha_T = (final_T/init_T)**(1.0/n_iter)    # Equation (2) | so alpha_T is c in the paper
    print("init_T       :", init_T)
    print("final_T      :", final_T)
    print("n_iter       :", n_iter)
    print("alpha_T      :", alpha_T)
    print("n_iter_fleet :", n_iter_fleet)
    
    coords = data[:, :2]
    print("Data Type")
    print(coords.shape, data.shape)
    distance_matrix = np.zeros([len(coords), len(coords)])
    for i in range(len(coords)):
        coord = coords[i]
        distance_matrix[i] = np.sum((coord-coords)**2, axis=1)**0.5 # euclidean distance
    
    if init_route is not None:
        best_routes = copy.deepcopy(init_route)
    else:
        print("No Initial Route")
        print("Generate Several Best Routes")
        print("Initial routes: There are {} numbers of routes".format(len(range(1, len(data)))))
        best_routes = [[i] for i in range(1, len(data))]
        print("Best Routes:\n", best_routes)
    print()
    print("Get First Routes Distance")
    best_distance = get_routes_distance(distance_matrix, best_routes, _print = False)
    last_routes = copy.deepcopy(best_routes)
    last_distance = get_routes_distance(distance_matrix, best_routes) # actually same as before
    neighbours = get_neighbours(distance_matrix)
    print(len(best_routes), best_distance)
    # plot_routes(data, best_routes)
    
#     if n_iter_fleet is None: n_iter_fleet=int(max(n_iter*0.1, 1))
    if (n_iter_fleet is not None) and (n_iter_fleet > 0):
        last_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, best_routes, verbose_step)
        last_distance = get_routes_distance(distance_matrix, last_routes)

        if last_distance < best_distance:
            best_distance = last_distance
            best_routes = last_routes
        print(len(best_routes), best_distance)
    
    temperature = init_T
    # the algorithm
    for i_iter in range(n_iter):
        if obj_n_routes is not None and len(last_routes) > obj_n_routes and (i_iter+1) % fleet_gap == 0:
            # not called it seems
            # current_routes = fleet_min(n_iter_fleet, data, distance_matrix, neighbours, last_routes, verbose_step)
            pass
        else:
            current_routes, absents = ruin(last_routes, neighbours)
            ruined_routes = copy.deepcopy(current_routes)
            current_routes = recreate(data, distance_matrix, current_routes, absents)
        
        current_distance = get_routes_distance(distance_matrix, current_routes)
        # Equation (4) for Acceptance Criterion
        if len(current_routes) < len(best_routes) or \
            ( current_distance < (last_distance - temperature*np.log(np.random.random())) and \
            len(current_routes) <= len(best_routes) ):
            # Check whether num. of current routes < num of best routes OR current distance < best distance
            # Basically update the best routes & distance IFF the total distance is shorter OR the number of routes is lesser
            if len(current_routes) < len(best_routes) or current_distance < best_distance:
                best_distance = current_distance
                best_routes = current_routes
                if test_obj is not None and best_distance < test_obj:
                    break
            last_distance = current_distance
            last_routes = current_routes
        # Equation (1)
        temperature *= alpha_T # Basically temperature = previous_temperature X c
        
        if verbose_step is not None and (i_iter+1) % verbose_step == 0:
            print("="*100)
            print(
                i_iter+1, np.round((i_iter+1)/n_iter*100, 4), "%:", 
                len(best_routes), len(last_routes), len(current_routes), best_distance, last_distance, current_distance
            )
            print("Iteration                : {}/{} ({}%)".format(i_iter+1, n_iter, np.round((i_iter+1)/n_iter*100, 4)))
            print("Current Routes           :")
            for idx, _route in enumerate(current_routes):
                print("{}\t(length={})\t(distance={:.4f}) : {}".format(idx, len(_route), get_route_distance(distance_matrix, _route), _route))
            print("Current Routes length    : {}".format(len(current_routes)))
            print("Current Distance         : {}".format(current_distance))
            print("-"*50)
            print("Last Routes length       : {}".format(len(last_routes)))
            print("Last Distance            : {}".format(last_distance))
            print("-"*50)
            print("Best Routes              :")
            for idx, _route in enumerate(best_routes):
                print("{}\t(length={})\t(distance={:.4f}) : {}".format(idx, len(_route), get_route_distance(distance_matrix, _route), _route))
                if soft_time_window:
                    route_load = []
                    route_capacity = 200
                    for _node in _route[:-1]:
                        route_capacity = route_capacity + int(data[_node][2]) - int(data[_node][3])
                        route_load.append(route_capacity)
                    print(f"\t(capacity={route_load})")
            print("Best Routes length       : {}".format(len(best_routes)))
            print("Best Distance            : {}".format(best_distance))
            print()
            
            if (i_iter+1) % (verbose_step*10) == 0:
                plot_routes(data, ruined_routes, title="Iteration: {} (Ruin)".format(i_iter+1), iteration=i_iter+1, ruin=True)
                plot_routes(data, best_routes, title="Iteration: {} (Recreate)".format(i_iter+1), iteration=i_iter+1)
    
    import imageio
    def create_animation_from_frames(output_folder='frames', animation_path='routes_animation.gif', fps=2):
        frames = []
        filenames = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
        for filename in filenames:
            frames.append(imageio.imread(os.path.join(output_folder, filename)))
        imageio.mimsave(animation_path, frames, fps=fps)
        
    # Create animation
    create_animation_from_frames()
    
    if verbose_step is not None and n_iter % verbose_step != 0: 
        print(i_iter+1, "100.0 %:", best_distance)
    return best_distance, best_routes

def plot_routes(data, routes, title=None, output_folder="frames", iteration=0, ruin=False):
    coords = data[:, :2]
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


