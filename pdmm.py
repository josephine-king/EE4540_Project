import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_error

# PDMM average consensus algorithm
def pdmm_average(adjacency_matrix, sensor_measurements, c, broadcast = False, num_iter = 5000, active_nodes = 1, transmission_loss = 0):
    
    n = adjacency_matrix.shape[0]
    # Initialize variables
    x = sensor_measurements.copy()
    x_true = np.mean(sensor_measurements) * np.ones(n)
    error_val = calculate_error(x, x_true)
    error_vals = [error_val]
    y = {}
    z = {}
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0:
                z[(i, j)] = np.random.normal(sensor_measurements[i], 1)

    iter = 0
    transmission = 0
    transmissions = [transmission]
    while iter < num_iter and error_val > 1e-14:
        if (active_nodes < 1):
            nodes = np.random.choice(range(n), int(active_nodes*n))
        else:
            nodes = range(n)

        for i in nodes:
            # x update
            x[i] = sensor_measurements[i]
            # Find neighbors
            neighbors_idx = np.where(adjacency_matrix[i, :] > 0)[0]
            # Loop through neighbors
            for j in neighbors_idx:
                A = 1 if i < j else -1
                x[i] -= A * z[(i, j)]
            x[i] /= (1 + c * len(neighbors_idx))

            # y update 
            for j in neighbors_idx: 
                A = 1 if i < j else -1
                y[(i, j)] = z[(i, j)] + 2 * c * A * x[i]

        # Update auxiliary variables
        z_prev = z.copy()
        for i in nodes:
            # With broadcast, transmission loss affects transmission of x_i to all neighbors
            if (broadcast == True and transmission_loss > 0):
                lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                if (lost == 1):
                    transmission += 1
                    continue
            neighbors_idx = np.where(adjacency_matrix[i, :] > 0)[0]
            for j in neighbors_idx:
                if (broadcast == False):
                    # With unicast, transmission loss only affects the transmission of y_ij to one neighbor
                    if (transmission_loss > 0):
                        lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                        if (lost == 1):
                            transmission += 1
                            continue
                    z[(j, i)] = y[(i, j)]
                    # Node specific transmission
                    transmission += 1
                else:
                    A = 1 if i < j else -1
                    z[(j, i)] = z_prev[(i, j)] + 2 * c * A * x[i]
            if (broadcast == True):
                # Common transmission
                transmission += 1
                
        # Calculate error
        error_val = calculate_error(x, x_true)
        error_vals.append(error_val)
        transmissions.append(transmission)

        iter += 1

    return error_vals, transmissions, x

def pdmm_median(adjacency_matrix, sensor_measurements, c, broadcast = False, num_iter = 5000, err=1e-14, active_nodes = 1, transmission_loss = 0):
    
    n = adjacency_matrix.shape[0]
    # Initialize variables
    x = sensor_measurements.copy()
    x_true = np.median(sensor_measurements) * np.ones(n)
    error_val = calculate_error(x, x_true)
    error_vals = [error_val]
    y = {}
    z = {}
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] > 0:
                z[(i, j)] = np.random.normal(sensor_measurements[i], 1)

    iter = 0
    transmission = 0
    transmissions = [transmission]
    while iter < num_iter and error_val > err:
 
        if (active_nodes < 1):
            nodes = np.random.choice(range(n), int(active_nodes*n))
        else:
            nodes = range(n)

        for i in nodes:
            # Find neighbors
            neighbors_idx = np.where(adjacency_matrix[i, :] > 0)[0]
            # Loop through neighbors
            sum = 0
            for j in neighbors_idx:
                A = 1 if i < j else -1
                sum += A * z[(i, j)]

            if (-1 - sum)/(c*len(neighbors_idx)) > sensor_measurements[i]:
                x[i] = (-1 - sum)/(c*len(neighbors_idx))
            elif (1 - sum)/(c*len(neighbors_idx)) < sensor_measurements[i]:
                x[i] = (1 - sum)/(c*len(neighbors_idx))
            else: 
                x[i] = sensor_measurements[i]

        # Update auxiliary variables
        z_prev = z.copy()
        for i in nodes:
            # With broadcast, transmission loss affects transmission of x_i to all neighbors
            if (broadcast == True and transmission_loss > 0):
                lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                if (lost == 1):
                    transmission += 1
                    continue
            neighbors_idx = np.where(adjacency_matrix[i, :] > 0)[0]
            for j in neighbors_idx:
                if (broadcast == True):
                    A = 1 if i < j else -1
                    z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x[i])
                else:
                    # With unicast, transmission loss only affects the transmission of y_ij to one neighbor
                    if (transmission_loss > 0):
                        lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                        if (lost == 1):
                            transmission += 1
                            continue
                    A = 1 if i < j else -1
                    z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x[i])
                    transmission += 1

            if (broadcast == True):
                # Common transmission
                transmission += 1
                
        # Calculate error
        error_val = calculate_error(x, x_true)
        error_vals.append(error_val)
        transmissions.append(transmission)

        iter += 1

    return error_vals, transmissions, x
