import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_global_rotations(local_quats, parents):
    """
    Computes global rotations from local rotations using Forward Kinematics.
    
    Args:
        local_quats: (N, J, 4) numpy array of quaternions (x, y, z, w).
        parents: List of length J, where parents[i] is the parent index of joint i.
                 -1 indicates the root. Must be topologically sorted.
                 
    Returns:
        global_quats: (N, J, 4) numpy array of global quaternions.
    """
    N, J, _ = local_quats.shape
    
    locals_per_joint = [R.from_quat(local_quats[:, j, :]) for j in range(J)]
    globals_per_joint = [None] * J
    
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            globals_per_joint[i] = locals_per_joint[i]
        else:
            # Child joint: Global = Parent_Global * Local
            globals_per_joint[i] = globals_per_joint[parent_idx] * locals_per_joint[i]
            
    global_quats = np.stack([r.as_quat() for r in globals_per_joint], axis=1)
    
    return global_quats

def compute_local_rotations(global_quats, parents):
    """
    Computes local rotations from global rotations using Inverse Kinematics algebra.
    
    Args:
        global_quats: (N, J, 4) numpy array.
        parents: List of length J.
        
    Returns:
        local_quats: (N, J, 4) numpy array.
    """
    N, J, _ = global_quats.shape
    
    globals_per_joint = [R.from_quat(global_quats[:, j, :]) for j in range(J)]
    locals_per_joint = [None] * J
    
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            locals_per_joint[i] = globals_per_joint[i]
        else:
            # Child: Local = inv(Parent_Global) * Child_Global
            parent_r = globals_per_joint[parent_idx]
            child_r = globals_per_joint[i]
            
            locals_per_joint[i] = parent_r.inv() * child_r
            
    local_quats = np.stack([r.as_quat() for r in locals_per_joint], axis=1)
    
    return local_quats

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_global_translations(global_rotations, local_offsets, parents):
    """
    Computes global translations (positions) given global rotations and local structural offsets.

    Args:
        global_rotations: (N, J, 4) numpy array of global quaternions (x, y, z, w).
                          These are the 'new' rotations you calculated previously.
        local_offsets: (J, 3) numpy array (or N, J, 3).
                       The local translation of each joint relative to its parent 
                       (often called bone offsets or bind pose translations).
        parents: List of length J with parent indices (-1 for root).

    Returns:
        global_translations: (N, J, 3) numpy array of global positions.
    """
    N, J, _ = global_rotations.shape
    
    # 1. Pre-convert all global quaternions to Scipy Rotation objects.
    list_of_rots = [R.from_quat(global_rotations[:, i, :]) for i in range(J)]
    
    # 2. Prepare output array
    global_translations = np.zeros((N, J, 3))
    
    # 3. Iterate through the kinematic tree
    for i, parent_idx in enumerate(parents):
        if parent_idx == -1:
            global_translations[:, i, :] = local_offsets[i]
        else:
            # Child Joint: Parent_Global_Pos + Parent_Global_Rot * Local_Offset
            parent_pos = global_translations[:, parent_idx, :]
            parent_rot = list_of_rots[parent_idx]
            
            offset_rotated = parent_rot.apply(local_offsets[i])
            
            global_translations[:, i, :] = parent_pos + offset_rotated
            
    return global_translations