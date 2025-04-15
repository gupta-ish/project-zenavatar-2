import numpy as np

def quat_xyzw_to_wxyz(q):
    """
    Convert a quaternion from XYZW format to WXYZ format.

    Parameters:
        q (array-like): A quaternion in XYZW format.

    Returns:
        np.ndarray: The quaternion in WXYZ format.
    """
    return np.array([q[3], q[0], q[1], q[2]])

def quat_wxyz_to_xyzw(q):
    """
    Convert a quaternion from WXYZ format to XYZW format.

    Parameters:
        q (array-like): A quaternion in WXYZ format.

    Returns:
        np.ndarray: The quaternion in XYZW format.
    """
    return np.array([q[1], q[2], q[3], q[0]])

def quaternion_to_rotation_matrix(q, w_first=True):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters:
        q (array-like): A quaternion [w, x, y, z] where:
                        - w is the scalar part
                        - x, y, z are the vector parts

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    if w_first:
        w, x, y, z = q
    else:
        x, y, z, w = q

    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),     1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x**2 + y**2)]
    ])

    return R

def skew_symmetric(p):
    """
    Generate a skew-symmetric matrix from a 3D vector.
    
    Parameters:
        p (array-like): A 3D vector (list, tuple, or NumPy array) of length 3.
    
    Returns:
        np.ndarray: A 3x3 skew-symmetric matrix.
    """
    if len(p) != 3:
        raise ValueError("Input vector must have exactly 3 elements.")
    return np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

def normalize(x, min_value, max_value, target_min=0.0, target_max=1.0):
    """
    Normalize a value from a given range to a target range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - min_value) / (max_value - min_value)

    # Scale the value to the target range
    scaled = target_min + normalized * (target_max - target_min)

    return scaled

def unnormalize(x, min_value, max_value, target_min=0.0, target_max=1.0):
    """
    Unnormalize a value from a target range to a given range.
    """
    # Normalize the value to the range [0, 1]
    normalized = (x - target_min) / (target_max - target_min)

    # Scale the value to the target range
    scaled = min_value + normalized * (max_value - min_value)

    return scaled

def quat_rotate_inverse_numpy(q, v):
    shape = q.shape
    # q_w corresponds to the scalar part of the quaternion
    q_w = q[:, 0]
    # q_vec corresponds to the vector part of the quaternion
    q_vec = q[:, 1:]

    # Calculate a
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]

    # Calculate b
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0

    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a - b + c

def quat_rotate_numpy(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    # Calculate c
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0

    return a + b + c

# Example usage
# q = [0.7071, 0.7071, 0, 0]  # Quaternion [w, x, y, z]
# R = quaternion_to_rotation_matrix(q)
# print("Rotation Matrix:")
# print(R)