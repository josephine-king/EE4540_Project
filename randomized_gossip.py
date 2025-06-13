import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from utils import calculate_error

# Get W_bar to check optimality of the improved algorithm
def get_W_bar(P, adjacency_matrix):
    n = adjacency_matrix.shape[0]
    W_bar = np.zeros_like(adjacency_matrix, dtype='float64')

    for i in range(0, n):
        for j in range(0, n):
            if (adjacency_matrix[i, j] > 0):
                e_i = np.zeros((n, 1)); e_i[i] = 1
                e_j = np.zeros((n, 1)); e_j[j] = 1
                W_ij = np.eye(n) - 0.5 * (e_i - e_j) @ (e_i - e_j).T
                W_bar += P[i, j] * W_ij
    
    return W_bar / n

def get_p_matrix(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    W_bar = cvx.Variable((n, n), symmetric=True)
    constraints = [W_bar >> 0]

    Wij_list = []
    for i in range(0, n):
        for j in range(0, n):
            if (adjacency_matrix[i, j] > 0):
                e_i = np.zeros((n, 1)); e_i[i] = 1
                e_j = np.zeros((n, 1)); e_j[j] = 1
                W_ij = np.eye(n) - 0.5 * (e_i - e_j) @ (e_i - e_j).T
                Wij_list.append(W_ij)
                # only add the W_ij corresponding to an existing edge

    Wij_flat = np.array([W.flatten() for W in Wij_list])
    # p - vector -  contains the nonzero entries in P
    p = cvx.Variable(len(Wij_list), nonneg=True)
    W_bar_vec = p @ Wij_flat / n  # CVXPY expression

    # Reshape back to matrix
    W_bar = cvx.reshape(W_bar_vec, (n, n), order='F')

    # Introduce the constraints for P  - > row sum has to be 1
    row_sums = [0] * n
    cnt = 0
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0:
                row_sums[i] += p[cnt]
                cnt += 1

    for i in range(n):
        constraints.append(row_sums[i] == 1)

    one_vec = np.ones((n, 1))
    # J = np.eye(n) - one_vec @ one_vec.T / n
    W_proj = W_bar - (one_vec @ one_vec.T)/n

    # Minimize the second largest eigenvalue of W_bar by minimizing the largest eigenvalue of W_proj
    obj = cvx.Minimize(cvx.lambda_max(W_proj))

    # define the cvx problem
    prob = cvx.Problem(obj, constraints)
    # SCS solver used for SDP
    prob.solve(solver=cvx.SCS)

    cnt = 0
    P = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if (adjacency_matrix[i, j] > 0):
                # .value we get the value from the cvx variable
                P[i, j] = p[cnt].value
                cnt += 1

    return np.round(P, 4)

# Implements the randomized gossip algorithm for the specified number of iterations
def randomized_gossip(adjacency_matrix, sensor_measurements, P, num_iter = 50000, transmission_loss = 0, calculate_Wk = False):
    num_sensors = adjacency_matrix.shape[0]
    true_average = np.mean(sensor_measurements) * np.ones(num_sensors)
    est_average = sensor_measurements.copy()
    # Initialization of variables
    error_val = calculate_error(est_average, true_average)
    iter  = 0
    error_vals = [error_val]

    Wk = np.identity(num_sensors)

    transmission = 0
    transmissions = [transmission]
    while iter < num_iter and error_val > 1e-14:
        # Consider transmission loss from i to j and j to i
        lost1 = 0
        lost2 = 0
        if (transmission_loss > 0):
            lost1 = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
            lost2 = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
        # Randomly choose a sensor
        i = np.random.randint(0, num_sensors)
        # Randomly choose a neighbor
        j = np.random.choice(len(P[i]), p=P[i]/sum(P[i]))
        # Compute the average of the two selected sensors
        average_val = 0.5*(est_average[i] + est_average[j])
        # est_average contains the estimates of the average for nodes
        if (lost1 == 0): 
            est_average[i] = average_val
        if (lost2 == 0):
            est_average[j] = average_val

        # Calculate the true W_k
        if (calculate_Wk):
            Wk_iter = np.eye(num_sensors)
            if (lost1 == 0):
                Wk_iter[i, i] = 0.5
                Wk_iter[i, j] = 0.5
            if (lost2 == 0):
                Wk_iter[j, j] = 0.5
                Wk_iter[j, i] = 0.5
            Wk = Wk @ Wk_iter

        # Update # of transmissions - both i and j transmit their values
        transmission += 2
        # Calculate the error
        error_val = calculate_error(est_average, true_average)
        error_vals.append(error_val)
        transmissions.append(transmission)
        iter += 1
    
    # Get the error between the true and desired W^k
    Wk_err_norm = np.linalg.norm(Wk - np.ones((num_sensors, num_sensors))/num_sensors)
    
    return error_vals, transmissions, est_average, Wk_err_norm