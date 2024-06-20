import numpy as np
import itertools

def sum(A, i, j):
    sum = 0.0
    for element in range(i, j):
        if A[element] != None:
            sum += A[element]
    return sum

def evalSpice(filename):
    try:
        with open(filename, 'r') as file:
            file = file.read()
    except FileNotFoundError:
        raise FileNotFoundError('Please give the name of a valid SPICE file as input')
    start = (file.find('.circuit')) + len('.circuit') 
    end = (file.find('.end'))
    if(start == -1 + len('.circuit') or end == -1):
        raise ValueError('Malformed circuit file')

    circuit_def = file[start: end]
    circuit_def = (circuit_def.strip()).split('\n')
    for branch in circuit_def:
        if branch[0] not in ['V', 'I', 'R']:
            raise ValueError('Only V, I, R elements are permitted')
    nodes = []
    count = 0

    for i in circuit_def:
        l = i.split()
        if l[1] not in nodes and l[1] == 'GND':
            nodes.append(l[1])
        if l[2] not in nodes and l[2] == 'GND':
            nodes.append(l[2])

    for i in circuit_def:
        l = i.split()
        if l[1] not in nodes:
            nodes.append(l[1])
        if l[2] not in nodes:
            nodes.append(l[2])

    num_nodes = len(nodes)
    max_variables = num_nodes*(num_nodes-1)

    node_def = {count: i for count, i in enumerate(nodes)}
    node_2 = {i: count for count, i in enumerate(nodes)}

    voltage = np.array([np.array([None for i in range(max_variables)]) for i in range(max_variables)])
    voltage_sources = {}
    voltage_count = 0
    for i in range(len(circuit_def)):
        if circuit_def[i][0] ==  'V':
            l = circuit_def[i].split()
            if (voltage[node_2[l[1]]][node_2[l[2]]]) == None or voltage[node_2[l[1]]][node_2[l[2]]] == float(l[4]):
                voltage_sources[l[0]] = [node_2[l[1]], node_2[l[2]]]
                voltage[node_2[l[1]]][node_2[l[2]]] = float(l[4])
                voltage[node_2[l[2]]][node_2[l[1]]] = -float(l[4])
                voltage_count += 1
            else:
                raise ValueError('Circuit error: no solution')

    current = np.array([np.array([None for i in range(max_variables)]) for i in range(max_variables)])
    # current_count = 0
    for i in range(len(circuit_def)):
        if circuit_def[i][0] ==  'I':
            l = circuit_def[i].split()
            current[node_2[l[1]]][node_2[l[2]]] = float(l[4])
            current[node_2[l[2]]][node_2[l[1]]] = -float(l[4])
            # current_count += 1

    resistor = np.array([np.zeros(max_variables) for i in range(max_variables)])
    # resistor_count = 0
    for i in range(len(circuit_def)):
        if circuit_def[i][0] ==  'R':
            l = circuit_def[i].split()
            if float(l[3]) != 0:
                resistor[node_2[l[1]]][node_2[l[2]]] = 1/float(l[3])
                resistor[node_2[l[2]]][node_2[l[1]]] = 1/float(l[3])

    # conductance_matrix = np.array([np.zeros(max_variables) for i in range(max_variables)])
    conductance_matrix = np.zeros((max_variables, max_variables))
    for i in range(num_nodes - 1):
        for j in range(num_nodes - 1):
            if i == j:
                conductance_matrix[i][j] = (np.sum(resistor, axis = 0))[i + 1]
                conductance_matrix[j][i] = (np.sum(resistor, axis = 0))[i + 1]
            else:
                conductance_matrix[i][j] = -resistor[i + 1][j + 1]
                # conductance_matrix[j][i] = -resistor[i + 1][j + 1]

    for i in range(num_nodes - 1):
        conductance_matrix[num_nodes - 1 + i][i] = conductance_matrix[i][i + num_nodes - 1] = 1

    combinations = list(itertools.combinations((range(num_nodes - 1)), 2))

    for i in range(len(combinations)):
        conductance_matrix[i + num_nodes - 1 + num_nodes - 1][combinations[i][0]] = -1
        conductance_matrix[i + num_nodes - 1 + num_nodes - 1][combinations[i][1]] = 1
        conductance_matrix[combinations[i][0]][i + num_nodes - 1 + num_nodes - 1] = -1
        conductance_matrix[combinations[i][1]][i + num_nodes - 1 + num_nodes - 1] = 1

    for j in range(max_variables - 1, -1, -1):
        if np.any(conductance_matrix[j, :]) == False:
            conductance_matrix = np.delete(conductance_matrix, j, axis= 0)
            conductance_matrix = np.delete(conductance_matrix, j, axis = -1)

    constant_matrix = np.zeros(len(conductance_matrix))

    for i in range(num_nodes - 1):
        constant_matrix[i] = sum(current[i + 1], 0, max_variables)

    j_1 = None
    j_2 = None

    for i in range(num_nodes - 1, 2*(num_nodes - 1)):
        constant_matrix[i] = voltage[i - num_nodes + 2][0]

    for i in range(2*(num_nodes - 1), len(conductance_matrix)):
        for j in range(len(conductance_matrix)):
            if conductance_matrix[i][j] == 1:
                j_1 = j
            elif conductance_matrix[i][j] == -1:
                j_2 = j
        constant_matrix[i] = voltage[j_1 + 1][j_2 + 1]

    # nan_mask = np.isnan(constant_matrix)
    for i in range(len(constant_matrix) - 1, -1, -1):
        if np.isnan(constant_matrix[i]):
            constant_matrix = np.delete(constant_matrix, i)
            conductance_matrix = np.delete(conductance_matrix, i, axis = 0)
            conductance_matrix = np.delete(conductance_matrix, i, axis = -1)

    try:
        X = np.linalg.solve(conductance_matrix, constant_matrix)
    except np.linalg.LinAlgError:
        raise ValueError('Circuit error: no solution')
    
    solved_voltage = {'GND': 0.0,}

    for i in range(num_nodes - 1):
        solved_voltage[node_def[i + 1]] = X[i]

    solved_current = {}
    j_1 = -1
    j_2 = -1
    for i in range(num_nodes - 1, len(conductance_matrix)):
        for j in range(len(conductance_matrix)):
            if conductance_matrix[i][j] == 1:
                j_1 = j
            elif conductance_matrix[i][j] == -1:
                j_2 = j
            for k in ((voltage_sources)):
                if ((voltage_sources[k])[0] == j_1 + 1 and (voltage_sources[k])[1]== j_2 + 1) or ((voltage_sources[k])[1] == j_1 + 1 and (voltage_sources[k])[0] == j_2 + 1):
                    if solved_voltage[node_def[(voltage_sources[k])[0]]] > solved_voltage[node_def[(voltage_sources[k])[1]]] and node_def[(voltage_sources[k])[0]] > node_def[(voltage_sources[k])[1]]:
                        solved_current[k] = -X[i] 
                    elif solved_voltage[node_def[(voltage_sources[k])[0]]] < solved_voltage[node_def[(voltage_sources[k])[1]]] and node_def[(voltage_sources[k])[0]] < node_def[(voltage_sources[k])[1]]:
                        solved_current[k] = -X[i] 
                    else:
                        solved_current[k] = X[i]

    return (solved_voltage, solved_current)