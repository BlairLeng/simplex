import numpy as np
 
class Simplex(object):
    def __init__(self, obj, max_mode=False):
        self.mat, self.max_mode = np.array([[0] + obj]) * (-1 if max_mode else 1), max_mode
 
    def add_constraint(self, a, b):
        self.mat = np.vstack([self.mat, [b] + a])
 
    def _simplex(self, mat, B, m, n):
        while mat[0, 1:].min() < 0:
            col = np.where(mat[0, 1:] < 0)[0][0] + 1 
            row = np.array([mat[i][0] / mat[i][col] if mat[i][col] > 0 else 0x7fffffff for i in
                            range(1, mat.shape[0])]).argmin() + 1  
            if mat[row][col] <= 0: return None  
            self._pivot(mat, B, row, col)
        return mat[0][0] * (1 if self.max_mode else -1), {B[i]: mat[i, 0] for i in range(1, m) if B[i] < n}
 
    def _pivot(self, mat, B, row, col):
        mat[row] /= mat[row][col]
        ids = np.arange(mat.shape[0]) != row
        mat[ids] -= mat[row] * mat[ids, col:col + 1]  
        B[row] = col
 
    def solve(self):
        m, n = self.mat.shape 
        temp, B = np.vstack([np.zeros((1, m - 1)), np.eye(m - 1)]), list(range(n - 1, n + m - 1))  
        mat = self.mat = np.hstack([self.mat, temp]) 
        if mat[1:, 0].min() < 0: 
            row = mat[1:, 0].argmin() + 1  
            temp, mat[0] = np.copy(mat[0]), 0  
            mat = np.hstack([mat, np.array([1] + [-1] * (m - 1)).reshape((-1, 1))])
            self._pivot(mat, B, row, mat.shape[1] - 1)
            if self._simplex(mat, B, m, n)[0] != 0: return None 
 
            if mat.shape[1] - 1 in B:
                self._pivot(mat, B, B.index(mat.shape[1] - 1), np.where(mat[0, 1:] != 0)[0][0] + 1)
            self.mat = np.vstack([temp, mat[1:, :-1]]) 
            for i, x in enumerate(B[1:]):
                self.mat[0] -= self.mat[0, x] * self.mat[i + 1]
        return self._simplex(self.mat, B, m, n)

"""
Here you can define your problem like this
Add constraint like this
Calling solve to solve the problem
"""
#minimize MIN -2x1 -12x2 -3x3
#x1 + x3 <= 1
#...

t = Simplex([-2, -12, -3])
t.add_constraint([1, 0, 1], 1)
t.add_constraint([0, 1, 0], 3)
t.add_constraint([0, 0, 1], 4)
t.add_constraint([1, 3, 1], 6)
print(t.solve())