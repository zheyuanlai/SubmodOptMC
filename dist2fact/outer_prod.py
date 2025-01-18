import numpy as np # type: ignore

def compute_outer_product(A, B):
    '''
    Computes the outer product of matrices A and B.

    Parameters:
        A (ndarray): matrix A
        B (ndarray): matrix B
    
    Returns:
        ndarray: The outer product matrix A ⊗ B.
    '''
    n_A = A.shape[0]
    n_B = B.shape[0]

    outer = np.zeros((n_A * n_B, n_A * n_B))

    for i in range(n_A):
        for j in range(n_A):
            for k in range(n_B):
                for l in range(n_B):
                    row_index = i * n_B + k
                    col_index = j * n_B + l
                    outer[row_index, col_index] = A[i, j] * B[k, l]
    
    return outer

# Example usage
if __name__ == "__main__":
    P_S = np.array([[0.6, 0.4],
                    [0.3, 0.7]])
    P_minus_S = np.array([[0.8, 0.2],
                          [0.5, 0.5]])
    
    P_outer = compute_outer_product(P_S, P_minus_S)
    print("Outer product matrix P_S ⊗ P_{-S}:")
    print(P_outer)
