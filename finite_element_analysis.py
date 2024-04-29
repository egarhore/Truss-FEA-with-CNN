import numpy as np


def assemble_stiffness_matrix(all_lines, line_nodes, node_number, geo_properties):
    # Extract the geometric properties
    t = geo_properties['thickness']
    r = geo_properties['radius']
    young_mod = geo_properties['youngModulus']

    # Calculate the area of the truss
    r1 = r - t
    t_area = (np.pi * r * r) - (np.pi * r1 * r1)

    # Calculate the second moment of area
    i_area = (np.pi * ((np.power(r, 4)) - (np.power(r1, 4)))) / 4

    # Get the number of elements (number of lines)
    num_lines = len(all_lines)

    # Declare arrays to contain the stiffness matrix and transformation matrix
    stiff_matrix = np.zeros([node_number, node_number])
    line_stiff_matrix = np.zeros([6, 6, num_lines])
    line_transform_matrix = np.zeros([6, 6, num_lines])

    # Loop over the number of elements creating the stiffness matrix
    for i in range(num_lines):
        k = np.zeros([6, 6])

        # Get the length of the trusses
        diff_x = all_lines[i, 2] - all_lines[i, 0]
        diff_y = all_lines[i, 3] - all_lines[i, 1]
        x_len = diff_x * diff_x
        y_len = diff_y * diff_y
        line_length = np.sqrt(x_len + y_len)

        # Calculate the cosine of theta as (x2 - x1)/l
        c_value = diff_x / line_length

        # Calculate the sin of theta as (y2 - y1)/l
        s_value = diff_y / line_length

        # Solve for the transformation matrix
        t_matrix = np.array([[c_value, s_value, 0, 0, 0, 0],
                             [-s_value, c_value, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, c_value, s_value, 0],
                             [0, 0, 0, -s_value, c_value, 0],
                             [0, 0, 0, 0, 0, 1]])

        # Obtain the node number for the element (truss) for global assembly
        n1 = line_nodes[i, 0] * 3
        n2 = line_nodes[i, 1] * 3

        # Assemble the local stiffness matrix
        k[0, 0] = (t_area * young_mod) / line_length
        k[0, 1] = 0
        k[0, 2] = 0
        k[0, 3] = -(t_area * young_mod) / line_length
        k[0, 4] = 0
        k[0, 5] = 0

        k[1, 0] = k[0, 1]
        k[1, 1] = (12 * young_mod * i_area) / (line_length ** 3)
        k[1, 2] = (6 * young_mod * i_area) / (line_length ** 2)
        k[1, 3] = 0
        k[1, 4] = -(12 * young_mod * i_area) / (line_length ** 3)
        k[1, 5] = (6 * young_mod * i_area) / (line_length ** 2)

        k[2, 0] = k[0, 2]
        k[2, 1] = k[1, 2]
        k[2, 2] = (4 * young_mod * i_area) / line_length
        k[2, 3] = 0
        k[2, 4] = -(6 * young_mod * i_area) / (line_length ** 2)
        k[2, 5] = (2 * young_mod * i_area) / line_length

        k[3, 0] = k[0, 3]
        k[3, 1] = k[1, 3]
        k[3, 2] = k[2, 3]
        k[3, 3] = (t_area * young_mod) / line_length
        k[3, 4] = 0
        k[3, 5] = 0

        k[4, 0] = k[0, 4]
        k[4, 1] = k[1, 4]
        k[4, 2] = k[2, 4]
        k[4, 3] = k[3, 4]
        k[4, 4] = (12 * young_mod * i_area) / (line_length ** 3)
        k[4, 5] = -(6 * young_mod * i_area) / (line_length ** 2)

        k[5, 0] = k[0, 5]
        k[5, 1] = k[1, 5]
        k[5, 2] = k[2, 5]
        k[5, 3] = k[3, 5]
        k[5, 4] = k[4, 5]
        k[5, 5] = (4 * young_mod * i_area) / line_length

        # Apply the transformation matrix
        k = np.matmul(np.transpose(t_matrix), k)
        k = np.matmul(k, t_matrix)

        # Add the local stiffness matrix to the global stiffness matrix
        stiff_matrix[n1, n1] = stiff_matrix[n1, n1] + k[0, 0]
        stiff_matrix[n1, n1 + 1] = stiff_matrix[n1, n1 + 1] + k[0, 1]
        stiff_matrix[n1, n1 + 2] = stiff_matrix[n1, n1 + 2] + k[0, 2]
        stiff_matrix[n1, n2] = stiff_matrix[n1, n2] + k[0, 3]
        stiff_matrix[n1, n2 + 1] = stiff_matrix[n1, n2 + 1] + k[0, 4]
        stiff_matrix[n1, n2 + 2] = stiff_matrix[n1, n2 + 2] + k[0, 5]

        stiff_matrix[n1 + 1, n1] = stiff_matrix[n1 + 1, n1] + k[1, 0]
        stiff_matrix[n1 + 1, n1 + 1] = stiff_matrix[n1 + 1, n1 + 1] + k[1, 1]
        stiff_matrix[n1 + 1, n1 + 2] = stiff_matrix[n1 + 1, n1 + 2] + k[1, 2]
        stiff_matrix[n1 + 1, n2] = stiff_matrix[n1 + 1, n2] + k[1, 3]
        stiff_matrix[n1 + 1, n2 + 1] = stiff_matrix[n1 + 1, n2 + 1] + k[1, 4]
        stiff_matrix[n1 + 1, n2 + 2] = stiff_matrix[n1 + 1, n2 + 2] + k[1, 5]

        stiff_matrix[n1 + 2, n1] = stiff_matrix[n1 + 2, n1] + k[2, 0]
        stiff_matrix[n1 + 2, n1 + 1] = stiff_matrix[n1 + 2, n1 + 1] + k[2, 1]
        stiff_matrix[n1 + 2, n1 + 2] = stiff_matrix[n1 + 2, n1 + 2] + k[2, 2]
        stiff_matrix[n1 + 2, n2] = stiff_matrix[n1 + 2, n2] + k[2, 3]
        stiff_matrix[n1 + 2, n2 + 1] = stiff_matrix[n1 + 2, n2 + 1] + k[2, 4]
        stiff_matrix[n1 + 2, n2 + 2] = stiff_matrix[n1 + 2, n2 + 2] + k[2, 5]

        stiff_matrix[n2, n1] = stiff_matrix[n2, n1] + k[3, 0]
        stiff_matrix[n2, n1 + 1] = stiff_matrix[n2, n1 + 1] + k[3, 1]
        stiff_matrix[n2, n1 + 2] = stiff_matrix[n2, n1 + 2] + k[3, 2]
        stiff_matrix[n2, n2] = stiff_matrix[n2, n2] + k[3, 3]
        stiff_matrix[n2, n2 + 1] = stiff_matrix[n2, n2 + 1] + k[3, 4]
        stiff_matrix[n2, n2 + 2] = stiff_matrix[n2, n2 + 2] + k[3, 5]

        stiff_matrix[n2 + 1, n1] = stiff_matrix[n2 + 1, n1] + k[4, 0]
        stiff_matrix[n2 + 1, n1 + 1] = stiff_matrix[n2 + 1, n1 + 1] + k[4, 1]
        stiff_matrix[n2 + 1, n1 + 2] = stiff_matrix[n2 + 1, n1 + 2] + k[4, 2]
        stiff_matrix[n2 + 1, n2] = stiff_matrix[n2 + 1, n2] + k[4, 3]
        stiff_matrix[n2 + 1, n2 + 1] = stiff_matrix[n2 + 1, n2 + 1] + k[4, 4]
        stiff_matrix[n2 + 1, n2 + 2] = stiff_matrix[n2 + 1, n2 + 2] + k[4, 5]

        stiff_matrix[n2 + 2, n1] = stiff_matrix[n2 + 2, n1] + k[5, 0]
        stiff_matrix[n2 + 2, n1 + 1] = stiff_matrix[n2 + 2, n1 + 1] + k[5, 1]
        stiff_matrix[n2 + 2, n1 + 2] = stiff_matrix[n2 + 2, n1 + 2] + k[5, 2]
        stiff_matrix[n2 + 2, n2] = stiff_matrix[n2 + 2, n2] + k[5, 3]
        stiff_matrix[n2 + 2, n2 + 1] = stiff_matrix[n2 + 2, n2 + 1] + k[5, 4]
        stiff_matrix[n2 + 2, n2 + 2] = stiff_matrix[n2 + 2, n2 + 2] + k[5, 5]

        # Compile the total local matrices
        line_stiff_matrix[:, :, i] = k
        line_transform_matrix[:, :, i] = t_matrix

    return stiff_matrix, line_stiff_matrix, line_transform_matrix


# Apply penalty to the load vector and load vector for calculation
def apply_penalty(stiffness_matrix, load_vector, boundary_matrix, num_nodes):
    # Get the penalty value
    penalty = np.max(stiffness_matrix)
    penalty = penalty * (10 ** 8)

    # Loop over the nodes adding penalty to nodes with boundary conditions
    for i in range(num_nodes):
        if boundary_matrix[i, 0] != 0:
            stiffness_matrix[i, i] = stiffness_matrix[i, i] + penalty
            load_vector[i] = load_vector[i] + (penalty * boundary_matrix[i, 1])
    return stiffness_matrix, load_vector


# Redistribute the displacement vector from the reduced size to the full scale
def redistribute_vector(disp_vector, reduced_disp_vector, removed_index, num_nodes):
    # Get the nodes to keep
    nodes = np.array(np.arange(0, num_nodes, 1), dtype=int)
    nodes = np.delete(nodes, obj=removed_index, axis=0)
    n = len(nodes)
    if n * 3 != len(reduced_disp_vector):
        print('Error redistributing displacement vector')
        return 0

    # Redistribute the displacement vector
    count = 0
    for i in range(len(nodes)):
        k = nodes[i] * 3
        m = count * 3
        disp_vector[k] = reduced_disp_vector[m]
        disp_vector[k + 1] = reduced_disp_vector[m + 1]
        disp_vector[k + 2] = reduced_disp_vector[m + 2]
        count = count + 1
    return disp_vector


# Calculate the stresses in each truss
def obtain_stresses(line_len, new_line_len, geo_properties):
    t = geo_properties['thickness']
    r = geo_properties['radius']
    young_mod = geo_properties['youngModulus']

    # Calculate change in length
    len_change = new_line_len - line_len

    # Calculate strain
    strain = np.divide(len_change, line_len)

    # Calculate the stresses
    stress = strain * young_mod

    # Calculate the area of the truss
    r1 = r - t
    t_area = (np.pi * r * r) - (np.pi * r1 * r1)

    # Calculate the force on each element
    element_f = stress * t_area

    return stress, element_f


# Calculate the reaction forces
def obtain_reaction_forces(all_k, all_t, all_lines, line_nodes, node_num, disp_vector):
    # Count the number of elements/lines (trusses)
    num_elem = len(all_lines)

    # Initialise the reaction force vector
    reaction_f = np.zeros([node_num, 1])

    # Loop over the number of element assembling the reaction forces
    for i in range(num_elem):
        # Get the nodes to change
        n1 = line_nodes[i, 0] * 3
        n2 = line_nodes[i, 1] * 3
        node_2_change = np.array([n1, n1 + 1, n1 + 2, n2, n2 + 1, n2 + 2], dtype=int)
        reaction_f[node_2_change] = \
            reaction_f[node_2_change] + np.matmul(all_k[:, :, i], np.matmul(all_t[:, :, i], disp_vector[node_2_change]))

    return reaction_f


# Calculate the strain energy
def get_strain_energy(reaction_f, disp_vector, num_nodes):
    # Get the forces
    x_force = reaction_f[np.array(np.arange(0, num_nodes, 3), dtype=int)]
    y_force = reaction_f[np.array(np.arange(1, num_nodes, 3), dtype=int)]
    m_force = reaction_f[np.array(np.arange(2, num_nodes, 3), dtype=int)]

    # Get the displacements
    x_disp = disp_vector[np.array(np.arange(0, num_nodes, 3), dtype=int)]
    y_disp = disp_vector[np.array(np.arange(1, num_nodes, 3), dtype=int)]
    m_disp = disp_vector[np.array(np.arange(2, num_nodes, 3), dtype=int)]

    # Solve for magnitude force and displacement
    mag_force = np.sqrt((x_force * x_force) + (y_force * y_force))
    mag_disp = np.sqrt((x_disp * x_disp) + (y_disp * y_disp))

    # Solve for strain energy
    strain_energy = 0.5 * mag_force * mag_disp
    strain_energy = strain_energy + np.abs(0.5 * m_force * m_disp)

    # Remove boundary/zeros values
    x_i = np.zeros([len(x_force), 1])
    y_i = np.zeros([len(x_force), 1])
    for i in range(len(x_force)):
        if x_force[i] < 0:
            x_i[i] = i
        if y_force[i] < 0:
            y_i[i] = i

    x_force = np.abs(x_force)
    y_force = np.abs(y_force)

    for i in range(len(x_force)):
        if x_force[i] < 10 ** -11:
            x_force[i] = 0
        if y_force[i] < 10 ** -8:
            y_force[i] = 0

    # Remove the zeros from the index list
    x_i = np.array([i for i in x_i if i != 0], dtype=int)
    y_i = np.array([i for i in y_i if i != 0], dtype=int)

    x_force[x_i] = x_force[x_i] * -1
    y_force[y_i] = y_force[y_i] * -1

    mag_force = np.sqrt((x_force * x_force) + (y_force * y_force))

    return strain_energy, mag_force


def solve_for_stiffness(u1, u2, l_nodes, f1, f2, mass_t):
    # Get the displacements and load
    x_disp = u1[l_nodes]
    y_disp = u2[l_nodes]
    f1 = f1[l_nodes]
    f2 = f2[l_nodes]

    # Calculate the resultant displacement and load
    r_disp = np.sqrt(np.multiply(x_disp, x_disp) + np.multiply(y_disp, y_disp))
    r_f = np.sqrt(np.multiply(f1, f1) + np.multiply(f2, f2))

    # Calculate the stiffness
    frame_stiff = np.abs(np.divide(r_f, r_disp))
    frame_stiff = np.sum(frame_stiff)
    # print(frame_stiff)
    # print(mass_t)
    frame_stiff = frame_stiff / mass_t

    return frame_stiff
