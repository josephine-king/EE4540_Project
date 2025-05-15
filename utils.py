import numpy as np
import matplotlib.pyplot as plt

# Given a communication range, field size, and dimension d, calculates 
# the number of agents such that the graph is connected with probability 1 - n^2
def calculate_num_agents(communication_range, field_size, d):
    norm_comm_range = communication_range / field_size
    num_agents = 2
    while norm_comm_range**d < 2*np.log(num_agents)/num_agents:
        num_agents += 1
    return num_agents

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Calculates the adjacency matrix, degree matrix, and Laplacian matrix of the graph
def calculate_graph_matrices(agent_locations, communication_range):
    num_agents = agent_locations.shape[1]
    adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)

    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            distance = np.linalg.norm(agent_locations[:,i] - agent_locations[:,j])
            if distance <= communication_range:
                adjacency_matrix[i,j] = 1
                adjacency_matrix[j,i] = 1   

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    return adjacency_matrix, degree_matrix, laplacian_matrix

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Calculates the eigenvalues of the Laplacian matrix to determine if the graph is connected
def is_graph_connected(laplacian_matrix):
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    zero_eigenvalues = np.sum(np.isclose(eigenvalues, 0))
    return zero_eigenvalues == 1

# Code adapted from the course "Probabilistic Sensor Fusion" 
# Visualizes the graph of sensor agents in the field
def visualize_graphs(agent_locations , adjacency_matrix):
    """
    Visualizes the underlying graph
    """
    num_agents = agent_locations.shape[1]
    colors = plt.cm.get_cmap("tab10", num_agents)  # Use a colormap with N unique colors

    # Plot the agents
    plt.figure(figsize=(6, 6))
    for i in range(num_agents):
        plt.scatter(agent_locations[0, i], agent_locations[1, i], marker = '.', s = 70, color='red')

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                plt.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1, alpha=0.2)
    
    plt.axis("equal")
    plt.show()
