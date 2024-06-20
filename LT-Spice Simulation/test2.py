import numpy as np
import itertools

def custom_sum(arr, start, end):
    return sum(val for val in arr[start:end] if val is not None)

def evalSpice(filename):
    try:
        with open(filename, 'r') as file:
            circuit_content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError('Please provide a valid SPICE file as input')

    try:
        start = circuit_content.find('.circuit') + len('.circuit')
        end = circuit_content.find('.end')
    except Exception as e:
        raise e

    circuit_def = circuit_content[start:end].strip().split('\n')

    for branch in circuit_def:
        if branch[0] not in ['V', 'I', 'R']:
            raise ValueError('Only V, I, R elements are permitted')

    nodes = set()
    nodes.add('GND')

    for line in circuit_def:
        parts = line.split()
        nodes.update(parts[1:3])

    num_nodes = len(nodes)
    max_variables = num_nodes * (num_nodes - 1)

    node_mapping = {}
    for idx, node in enumerate(nodes):
        node_mapping[idx] = node

    voltage = np.full((max_variables, max_variables), None, dtype=object)
    voltage_sources = {}
    voltage_count = 0

    for line in circuit_def:
        parts = line.split()
        if parts[0] == 'V':
            node1, node2, _, voltage_val = parts
            node1_idx, node2_idx = node_mapping[node1], node_mapping[node2]
            voltage_idx = (node1_idx, node2_idx)
            
            if voltage[voltage_idx] is None or voltage[voltage_idx] == float(voltage_val):
                voltage_sources[parts[0]] = voltage_idx
                voltage[voltage_idx] = float(voltage_val)
                voltage[node2_idx][node1_idx] = -float(voltage_val)
                voltage_count += 1
            else:
                raise ValueError('Circuit error: no solution')

    current = np.full((max_variables, max_variables), None, dtype=object)
    current_count = 0

    for line in circuit_def:
        parts = line.split()
        if parts[0] == 'I':
            node1, node2, _, current_val = parts
            node1_idx, node2_idx = node_mapping[node1], node_mapping[node2]
            current[node1_idx][node2_idx] = float(current_val)
            current[node2_idx][node1_idx] = -float(current_val)
            current_count += 1

    resistor = np.zeros((max_variables, max_variables))
    resistor_count = 0

    for line in circuit_def:
        parts = line.split()
        if parts[0] == 'R':
            node1, node2, _, resistance_val = parts
            resistance_val = float(resistance_val)
            if resistance_val != 0:
                node1_idx, node2_idx = node_mapping[node1], node_mapping[node2]
                resistor[node1_idx][node2_idx] = 1 / resistance_val
                resistor[node2_idx][node1_idx] = 1 / resistance_val

    conductance_matrix = np.zeros((max_variables, max_variables))

    for i in range(num_nodes - 1):
        conductance_matrix[i][i] = np.sum(resistor, axis=0)[i + 1]
        conductance_matrix[i][i + num_nodes - 1] = 1
        conductance_matrix[i + num_nodes - 1][i] = 1

    combinations = list(itertools.combinations(range(num_nodes - 1), 2))

    for i, combination in enumerate(combinations):
        conductance_matrix[i + num_nodes - 1 + num_nodes - 1][combination[0]] = -1
        conductance_matrix[i + num_nodes - 1 + num_nodes - 1][combination[1]] = 1
        conductance_matrix[combination[0]][i + num_nodes - 1 + num_nodes - 1] = -1
        conductance_matrix[combination[1]][i + num_nodes - 1 + num_nodes - 1] = 1

    # Remove rows/columns with all zeros
    non_zero_rows = np.any(conductance_matrix, axis=1)
    conductance_matrix = conductance_matrix[non_zero_rows][:, non_zero_rows]

    constant_matrix = np.zeros(len(conductance_matrix))

    for i in range(num_nodes - 1):
        constant_matrix[i] = custom_sum(current[i + 1], 0, max_variables)

    for i in range(num_nodes - 1, 2 * (num_nodes - 1)):
        constant_matrix[i] = voltage[i - num_nodes + 2][0]

    for i in range(2 * (num_nodes - 1), len(conductance_matrix)):
        for j in range(len(conductance_matrix)):
            if conductance_matrix[i][j] == 1:
                j_1 = j
            elif conductance_matrix[i][j] == -1:
                j_2 = j
        constant_matrix[i] = voltage[j_1 + 1][j_2 + 1]

    nan_mask = np.isnan(constant_matrix)
    constant_matrix = constant_matrix[~nan_mask]
    conductance_matrix = conductance_matrix[~nan_mask][:, ~nan_mask]

    try:
        X = np.linalg.solve(conductance_matrix, constant_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('Circuit error: no solution')

    solved_voltage = {'GND': 0.0}

    for i in range(num_nodes - 1):
        solved_voltage[node_mapping[i + 1]] = X[i]

    solved_current = {}

    for i in range(num_nodes - 1, len(conductance_matrix)):
        for j in range(len(conductance_matrix)):
            if conductance_matrix[i][j] == 1:
                j_1 = j
            elif conductance_matrix[i][j] == -1:
                j_2 = j
        for k in voltage_sources:
            source_node1, source_node2 = voltage_sources[k]
            if (
                (source_node1 == j_1 + 1 and source_node2 == j_2 + 1) or
                (source_node1 == j_2 + 1 and source_node2 == j_1 + 1)
            ):
                if (
                    solved_voltage[node_mapping[source_node1]] > solved_voltage[node_mapping[source_node2]] and
                    node_mapping[source_node1] > node_mapping[source_node2]
                ):
                    solved_current[k] = -X[i]
                elif (
                    solved_voltage[node_mapping[source_node1]] < solved_voltage[node_mapping[source_node2]] and
                    node_mapping[source_node1] < node_mapping[source_node2]
                ):
                    solved_current[k] = -X[i]
                else:
                    solved_current[k] = X[i]

    return (solved_voltage, solved_current)

# Example usage:
solved_voltage, solved_current = evalSpice('test_1.ckt')
print