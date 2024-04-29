import datetime
import time

import numpy as np
import numpy.linalg as lin_np
from constraintEquations import check_intersection, check_connection, check_joint_count
from basicFunctions import create_geometry, set_load_bound, create_connectivity, plot_connected_structures
from basicFunctions import solve_line_length_angle, connectivity_reduction, reduced_to_full, plot_nodes
from basicFunctions import plot_cnn_prediction
from finite_element_analysis import assemble_stiffness_matrix, apply_penalty, redistribute_vector, obtain_stresses
from finite_element_analysis import obtain_reaction_forces, get_strain_energy, solve_for_stiffness
from cnn_analysis import cnn_prediction

# Start the timer
start_time = time.time()

# Get size of the structure
totalLength = int(input('Enter length of the structure [mm] (Default value = 200): ') or 200)
totalHeight = int(input('Enter height of the structure [mm] (Default value = 200): ') or 200)
meshSize = int(input('Enter the partition size for the structure [mm] (Default value = 100): ') or 100)

# Get the material properties
E = float(input('Enter the young modulus of the material in use [Pa] (Default value = 1.61E11): ') or 161000000000)
r = float(input('Enter the radius of the trusses [m] (Default value = 0.005): ') or 0.005)
t = float(input('Enter the thickness of the trusses [m] (Default value = 0.0005): ') or 0.0005)
p = float(input('Enter the density of the truss material [kg/m^3] (Default value = 1350): ') or 1350)

# Created dictionary to save the material properties (very unnecessary)
geoProperties = {'youngModulus': E, 'radius': r, 'thickness': t, 'density': p}

# Get the load and boundary condition
bType = int(input('Enter the boundary type (Default value = 1): ') or 1)
lType = int(input('Enter the load type (Default value = 1): ') or 1)
theLoad = int(input('Enter the load (Default value = 1000): ') or 1000)

# Create the geometry of the structure
coords, nY, nX = create_geometry(totalLength, totalHeight, meshSize, bType)

# Get the total number of nodes and variables
nodeNumber = nY * nX

# Get the load and boundary condition
bcNodes, bcVector, lNodes, lVector = set_load_bound(nX, nY, bType, lType, theLoad)

# Obtain input and connectivity matrix and lines
to_plot = input('Do you want to view the nodes and their corresponding number? Yes or No (Default: No): ' or 'No')
if to_plot.lower() == 'yes':
    plot_nodes(coords)
numConns = int(input('Enter the number of connections: '))
print('Example: To connect node 1 to node 2 and node 0 to node 9 enter: 1 2 0 9')
conns = list(map(int, input('Enter first and second connected nodes continuously: ').strip().split()))[:numConns * 2]
conns = np.array(conns, dtype=int)
conns = conns.reshape(numConns, 2)

print('----------------------------------------------------------------------------------------')
print('Solving...')
print('----------------------------------------------------------------------------------------')

conns_input = conns
conns, all_lines, line_nodes = create_connectivity(nX, nY, conns, coords)

# Ensure the design meets the requirement for analysis
if check_intersection(all_lines):
    exit()
if check_connection(conns, lNodes, bcNodes):
    exit()
if check_joint_count(conns):
    exit()

# Solve for the length of all lines, angles of intersecting trusses and their masses
lineLength, lineAngle, lineMass = solve_line_length_angle(all_lines, geoProperties)
# Get the total mass of the structure (not including joints)
totalMass = np.sum(lineMass)

# Assemble stiffness matrix
nodeNumber_t = nodeNumber * 3
stiff_matrix, line_stiff_matrix, line_transform_matrix = \
    assemble_stiffness_matrix(all_lines, line_nodes, nodeNumber_t, geoProperties)

# Reduce the stiffness matricx, load vector and boundary matrix
reduced_conns, removed_index = connectivity_reduction(conns)
full_index = reduced_to_full(removed_index)
reduced_stiffness = np.delete(stiff_matrix, obj=full_index, axis=0)
reduced_stiffness = np.delete(reduced_stiffness, obj=full_index, axis=1)
reduced_lVector = np.delete(lVector, obj=full_index, axis=0)
reduced_bcVector = np.delete(bcVector, obj=full_index, axis=0)

# Apply penalty
num_nodes = len(reduced_lVector)
reduced_stiffness, reduced_lVector = apply_penalty(reduced_stiffness, reduced_lVector, reduced_bcVector, num_nodes)

# Solve for the displacement vector
reduced_dispVector = np.matmul(lin_np.inv(reduced_stiffness), reduced_lVector)

# Redistribute the displacement vector (set it to match the original size)
dispVector = np.zeros([nodeNumber_t, 1])
dispVector = redistribute_vector(dispVector, reduced_dispVector, removed_index, nodeNumber)

# Get the displacement in the x, y and theta direction
U1 = dispVector[np.array(np.arange(0, nodeNumber_t, 3), dtype=int)]
U2 = dispVector[np.array(np.arange(1, nodeNumber_t, 3), dtype=int)]
U3 = dispVector[np.array(np.arange(2, nodeNumber_t, 3), dtype=int)]

# Get the forces in the x, y and theta direction
F1 = lVector[np.array(np.arange(0, nodeNumber_t, 3), dtype=int)]
F2 = lVector[np.array(np.arange(1, nodeNumber_t, 3), dtype=int)]

# Calculate the stresses
coords_change = np.concatenate((U1, U2), axis=1)
coords_change = coords + coords_change
conns, new_all_lines, line_nodes = create_connectivity(nX, nY, conns_input, coords_change)
new_lineLength, new_lineAngle, new_lineMass = solve_line_length_angle(new_all_lines, geoProperties)
stress, element_f = obtain_stresses(lineLength, new_lineLength, geoProperties)

# Calculate the reaction forces
reaction_F =\
    obtain_reaction_forces(line_stiff_matrix, line_transform_matrix, all_lines, line_nodes, nodeNumber_t, dispVector)

# Calculate the strain energy
strain_energy, mag_Force = get_strain_energy(reaction_F, dispVector, nodeNumber_t)

# Perform the CNN prediction of the joints by first creating the images then performing the predictions
node_Recorder, joint_Avail, is_Good, percent_SE, joint_Mass = \
    cnn_prediction(all_lines, line_nodes, mag_Force, element_f, strain_energy)

# Calculate and display the specific stiffness of the structure
if len(removed_index) == nodeNumber:
    print('Error in connectivity\nStructure not fully connected...')
elif joint_Avail:
    specific_stiffness = solve_for_stiffness(U1, U2, lNodes, F1, F2, joint_Mass + totalMass)
    specific_stiffness = '{:.4e}'.format(specific_stiffness)
    if is_Good:
        print('----------------------------------------------------------------------')
        print('AI predicts a stable structure\nThere are no failed joints')
        print('Specific stiffness = ' + str(specific_stiffness) + ' N/kg')
        print('----------------------------------------------------------------------')
    else:
        print('----------------------------------------------------------------------')
        print('Failure in joint detected')
        print('Specific stiffness = ' + str(specific_stiffness) + ' N/kg')
        print('----------------------------------------------------------------------')
else:
    specific_stiffness = solve_for_stiffness(U1, U2, lNodes, F1, F2, totalMass)
    specific_stiffness = '{:.4e}'.format(specific_stiffness)
    print('--------------------------------------------------------------------------')
    print('No detected joint in the structure')
    print('Specific stiffness = ' + str(specific_stiffness) + ' N/kg')
    print('--------------------------------------------------------------------------')

# Complete process and show runtime
end_time = time.time()
run_time = end_time - start_time
run_time = str(datetime.timedelta(seconds=int(run_time)))
print('Process completed\n' + 'Total runtime = ' + run_time)
print('##############################################################################')

# Plot the structure
to_plot = input('Do you want a visualisation of the deformation? Yes or No (Default: No): ' or 'No')
if to_plot.lower() == 'yes':
    plot_connected_structures(coords, line_nodes, all_lines, dispVector)
# plot_connected_structures(coords, line_nodes, all_lines, 0)
to_plot = input('Do you want a visualisation of the CNN prediction? Yes or No (Default: No): ' or 'No')
if to_plot.lower() == 'yes':
    plot_cnn_prediction(coords, all_lines, node_Recorder, percent_SE, meshSize)
