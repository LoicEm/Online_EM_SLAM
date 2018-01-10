import numpy as np


def generate_landmarks(n, map_shape=(100, 100)):
    """
    Generate the coordinates of n lamndmarks at random, following a random uniform distribution centered on 0.
    Return a matrix of shape (n, 2) with x coordinates as first column and y coordinates as second.
        n : number of landmarks
        map_shape : tuple for the size of the map
    """
    map_shape = np.array(map_shape)
    return np.random.uniform(low=-map_shape/2, high=map_shape/2, size=(n, 2))


def generate_path(controls, f, Q, position=np.array([0,0,0]), **fkwargs):
    """
    Generate the path taken by the robot from given starting position, controls and an f function.
    Return a (n_iter, 3) matrix with the position of the robot for each iteration.
        n_iter : number of iterations
        controls : (n_iter, 2) array, each column giving respectively velocity and direction
        f : function used to get the new position. Must take the arguments(x_t, u_t) and any other keywords arguments
        Q : covariance matrix to get the noisy controls
        position: starting_position
        fkwargs: additional keyword arguments for f
    """
    path = [position]
    for c in controls:
        # Add noise to the controls
        noisy_controls = np.random.multivariate_normal(c, Q)
        # Determine the new position
        position = f(position, noisy_controls, **fkwargs)
        path.append(position)
    return np.array(path)

def f_paper(x_previous, u_t, dt=1, B=1.5):
    """Implementation of the path in 4.1"""
    new_angle = x_previous[2] + u_t[1]
    change = np.array([np.cos(new_angle),
                       np.sin(new_angle),
                       np.sin(u_t[1])/B]) * u_t[0] * dt
    return x_previous + change