class SISR(object):
    def __init__(self, lp, data, T_init, T_final, N_iter=10, initial_solution=None):
        super().__init__(**lp.__dict__)
        self.T_init = ...
        self.N_iter = N_iter
        self.data = data
        self.distance_matrix = self._compute_distance_matrix(data)
        self.initial_solution = initial_solution
        self.best_solution = None
        
    def _compute_distance_matrix(self, data):
        return ...