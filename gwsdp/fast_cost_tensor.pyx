import numpy as np
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.float32_t DTYPE_t

# Generate loss matrix L (which is a flatten tensor of n^2 \times n^2)
def cost_tensor(cnp.ndarray D1, cnp.ndarray D2):
    # assert D1.dtype == DTYPE and D2.dtype == DTYPE    
    # Need to write test case check for this one
    cdef int m = D1.shape[0]
    cdef int n = D2.shape[0]
    cdef int i, j, k, l
    cdef cnp.ndarray cost = np.empty((m, n, m, n))#, dtype=DTYPE)
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    cost[i, j, k, l] = (D1[i, k] - D2[j, l]) ** 2
    return cost
