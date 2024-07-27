def get_neighbours(distance_matrix):
    n_vertices = distance_matrix.shape[0] # assumed NxN matrix
    neighbours = []
    for i in range(n_vertices):
        index_dist = [(j, distance_matrix[i][j]) for j in range(n_vertices)]
        # [(0, dm[0][0]), (1, dm[0][1]), (2, dm[0][2]), (3, dm[0][3])]
        sorted_index_dist = sorted(index_dist, key=lambda x: x[1])
        # [(3, dm[0][3]), (1, dm[0][1]), (0, dm[0][0]), (2, dm[0][2])]
        neighbours.append([x[0] for x in sorted_index_dist])
        # [3, 1, 0, 2]
        # so this reads, for every vertex, get the neighbors sorted by the nearest distance
    # output is something like
    return neighbours


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    # non-symmetric distance between two nodes
    dm = np.random.randint(0, 20, size = [4, 4])
    # symmetric distance between two nodes
    dm = (dm + dm.T)/2
    dm = np.asarray(dm, dtype=np.int16)
    np.fill_diagonal(dm, 0) # in-place operation
    df_dm = pd.DataFrame(dm, index=range(len(dm)), columns=range(len(dm)))
    print(df_dm)
    print(get_neighbours(dm))