import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Given a communication range, field size, and dimension d, calculates 
# the number of sensors (n) such that the graph is connected with probability 1 - 1 / n^2
def calculate_num_sensors(communication_range, field_size, d):
    norm_comm_range = communication_range / field_size
    num_sensors = 2
    while norm_comm_range**d < 2*np.log(num_sensors)/num_sensors:
        num_sensors += 1
    return num_sensors

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Calculates the adjacency matrix, degree matrix, and Laplacian matrix of the graph
def calculate_graph_matrices(sensor_locations, communication_range):
    num_sensors = sensor_locations.shape[1]
    adjacency_matrix = np.zeros((num_sensors, num_sensors), dtype=int)

    for i in range(num_sensors):
        for j in range(num_sensors):
            if i == j:
                continue
            distance = np.linalg.norm(sensor_locations[:,i] - sensor_locations[:,j])
            if distance <= communication_range:
                adjacency_matrix[i,j] = 1

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    return adjacency_matrix, degree_matrix, laplacian_matrix

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Calculates the eigenvalues of the Laplacian matrix to determine if the graph is connected
def is_graph_connected(laplacian_matrix):
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    n_zero_eigenvalues = np.sum(np.isclose(eigenvalues, 0))
    return n_zero_eigenvalues == 1

def get_field_function():
    np.random.seed(10)
    params_x = np.random.uniform(-1e-5, 1e-5, 3)
    params_y = np.random.uniform(-2e-5, 2e-5, 3)
    bias = np.random.uniform(18, 25)
    def field_function(x, y):
        x_term = params_x[0] * x**3 + params_x[1] * x**2 + params_x[2] * x
        y_term = params_y[0] * y**3 + params_y[1] * y**2 + params_y[2] * y
        return x_term + y_term + bias
    return field_function

def get_sensor_measurements(sensor_locations, field_function, noise_std):
    # vectorize: func(x, y)     ->      func([x], [y])
    #            f(x,y) = v     ->      f([x], [y]) -> [v]
    vectorized_func = np.vectorize(field_function)
    measurements = vectorized_func(sensor_locations[0, :], sensor_locations[1, :])
    # draw measurements from standard Gaussian dist
    noise = np.random.normal(0, noise_std, size=measurements.shape)
    return measurements + noise

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Visualizes the graph of sensors in the field
def visualize_graphs(sensor_locations, adjacency_matrix, field_function, sensor_values):
    """
    Visualizes the underlying graph
    """
    num_sensors = sensor_locations.shape[1]

    plt.rc('axes', labelsize=16)
    fig = plt.figure(figsize=(7,7))

    # Plot the field values
    x1_vals = np.linspace(0, 100, 100)
    x2_vals = np.linspace(0, 100, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.array([[field_function(x1, x2) for x1, x2 in zip(row_x1, row_x2)]
              for row_x1, row_x2 in zip(X1, X2)])
    field_plot = plt.contourf(X1, X2, Z, levels=100, cmap='plasma', alpha=0.7, antialiased=True)

    # Plot edges
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            if adjacency_matrix[i, j] == 1:
                plt.plot([sensor_locations[0, i], sensor_locations[0, j]],
                         [sensor_locations[1, i], sensor_locations[1, j]],
                         'black', linewidth=1, zorder=1, alpha=0.2)
             
    # Plot the sensors
    norm = plt.Normalize(Z.min(), Z.max())
    for i in range(num_sensors):
        plt.scatter(sensor_locations[0, i], sensor_locations[1, i], marker = 'o', s = 130, c=sensor_values[i], cmap='plasma', norm=norm, edgecolors='black', linewidths=0.5)
        #plt.annotate(f'{i}', xy=(sensor_locations[0, i], sensor_locations[1, i]), fontsize=6, horizontalalignment='center', verticalalignment='center')

    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.rcParams['savefig.dpi'] = 600

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="8%", pad=0.2)
    plt.colorbar(field_plot, label='Temperature (Celsius)', location="right", cax=cax)

    plt.axis("equal")
    plt.show()
    fig.savefig("figures/graph.png", bbox_inches='tight')

def calculate_error(x_pred, x_true):
    return np.linalg.norm(x_pred - x_true)/x_true[0]

def calculate_error_median(x_pred, x_true_min, x_true_max):
    inside_bounds = (x_pred >= x_true_min) & (x_pred <= x_true_max)
    err = np.where(inside_bounds, 0, np.minimum(np.abs(x_pred - x_true_min), np.abs(x_pred - x_true_max)))
    return (np.linalg.norm(err)+np.var(x_pred))/(0.5*(x_true_min[0] + x_true_max[0]))

def make_tl_plots(error_vals, transmissions, title, filename):
    labels = ["No", "25%", "50%", "75%"]
    fig = plt.figure()
    for i in range(len(transmissions)):
        plt.semilogy(transmissions[i], error_vals[i], label=labels[i] + " Transmission Loss")
    plt.grid(True)
    plt.xlabel("Transmissions")
    plt.ylabel("Normalized Error")
    plt.title(title)
    plt.legend()
    fig.savefig("figures/" + filename + ".png", bbox_inches='tight')

def make_async_plots(error_vals, transmissions, title, filename):
    labels = ["100%", "25%", "50%", "75%"]
    fig = plt.figure()
    for i in range(len(transmissions)):
        plt.semilogy(transmissions[i], error_vals[i], label=labels[i] + " Active Nodes")
    plt.grid(True)
    plt.xlabel("Transmissions")
    plt.ylabel("Normalized Error")
    plt.title(title)
    plt.legend()
    fig.savefig("figures/" + filename + ".png", bbox_inches='tight')