import numpy as np

# ===========================================================================================================
# ===================================== Defining few functions for ease ====================================
# ==========================================================================================================

# --------- Conjugate Transpose --------

def conjT(A # Matrix 
         ):
    B = np.conj(A)
    B = np.transpose(B)
    return B

# --- Vector norm and Vector product

def vec_norm(v # column vector
            ):
    norm = conjT(v)@v
    return np.sqrt(norm[0,0])

def vec_prod(u,v # column vectors
            ):
    norm = conjT(u)@v
    return norm[0,0]


