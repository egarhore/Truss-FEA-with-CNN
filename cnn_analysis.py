import numpy as np
from basicFunctions import find_position
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json
from pathlib import Path
import pickle


def cnn_prediction(all_lines, line_nodes, mag_force, f_el, s_e):
    # Set the structure as stable
    is_good = True
    # Set the percentage difference of strain energy to 0
    percent_se = 0
    # Determine the number of nodes with trusses connected to them
    num_joints = np.unique(line_nodes)
    frame_joint = num_joints
    # Count the number of joints
    num_joints = len(num_joints)
    # Create a vector with the number of nodes as zeros
    node_recorder = []
    # Create a vector with the number of lines in the joint
    node_lines = np.zeros([num_joints, 5]) - 1
    # Create a vector to store the maximum forces acting on the joint
    joint_force = []
    # Create the array to store the images
    image_array = []
    # Initialise joint mass
    j_mass = []
    # Create an array to mark the joints to be removed
    to_remove = []
    # Set the joint availability to false
    joint_avail = False

    # Loop of the joints creating the images
    for k in range(num_joints):
        j = 0
        delta_x = 0
        delta_y = 0
        i = frame_joint[k]
        node_row, node_col = find_position(i, line_nodes)
        if len(node_row) > 1:
            # Create an array to save the angles
            theta = np.zeros([len(node_row), 1])
            # Declare the size of the forces
            d_force = np.zeros([len(node_row), 1])
            # Loop over the number of lines (trusses) connected to the node
            d_image, ax = plt.subplots()
            for j in range(len(node_row)):
                # Add the number of lines connected to the node
                node_lines[k, j] = node_row[j]
                # Get the node numbers
                node_1 = line_nodes[node_row[j], 0]
                node_2 = line_nodes[node_row[j], 1]
                # Get the forces
                f_node_1 = mag_force[node_1]
                f_node_2 = mag_force[node_2]
                # Get the force acting on the line (truss)
                d_force[j] = np.abs(f_el[node_row[j]])
                # Obtain the x and y coordinate of the line (truss)
                x = np.array([all_lines[node_row[j], 0], all_lines[node_row[j], 2]])
                y = np.array([all_lines[node_row[j], 1], all_lines[node_row[j], 3]])
                # Check the direction of line and flip if its in reverse
                if node_col[j] == 1:
                    x = np.flip(x)
                    y = np.flip(y)

                # determine the change in the coordinates
                delta_x = x[1] - x[0]
                delta_y = y[1] - y[0]

                # Determine the angle of the line (truss)
                theta[j] = delta_y / np.sqrt((delta_x * delta_x) + (delta_y * delta_y))
                theta[j] = np.arcsin(theta[j])

                # Determine the length of the line (truss)
                line_length = np.sqrt((delta_x * delta_x) + (delta_y * delta_y))

                # Determine the unit length
                unit_x = delta_x / line_length
                unit_y = delta_y / line_length

                # Update the coordinate to make each line length 6 to match training data
                x[1] = x[0] + (6 * unit_x)
                y[1] = y[0] + (6 * unit_y)

                # Shift the line to the origin [0, 0]
                x[1] = x[1] - x[0]
                x[0] = x[0] - x[0]
                y[1] = y[1] - y[0]
                y[0] = y[0] - y[0]

                # Plot the truss to the image
                if f_node_1 != 0 and f_node_2 != 0:
                    if f_el[node_row[j]] > 0 and node_col[j] == 0:
                        ax.plot(x, y, color='green', linewidth=3)
                    elif f_el[node_row[j]] > 0 and node_col[j] == 1:
                        ax.plot(x, y, color='blue', linewidth=3)
                    elif f_el[node_row[j]] < 0 and node_col[j] == 0:
                        ax.plot(x, y, color='blue', linewidth=3)
                    elif f_el[node_row[j]] < 0 and node_col[j] == 1:
                        ax.plot(x, y, color='green', linewidth=3)
                else:
                    ax.plot(x, y, color='red', linewidth=3)
            ax.axis([-6, 6, -6, 6])
            ax.axis('off')
            ax.margins(0)
            d_image.tight_layout()
            d_image.set_figwidth(0.64)
            d_image.set_figheight(0.64)
            d_image.set_dpi(100)

            # Convert plot to figure
            fig = plt.figure(d_image)
            # Grab the canvas of the figure
            fig.canvas.draw()
            # Save it to a numpy array
            j_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            j_image = j_image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

            # Check if all the lines have the same angle of intercept (means not a joint)
            theta = np.round(theta, 3)
            theta = np.rad2deg(theta)
            # Check the difference in the angle
            temp_theta = np.abs(np.abs(theta) - np.abs(theta[0]))
            logical_test = np.zeros([len(temp_theta), 1])
            for lc in range(len(temp_theta)):
                if temp_theta[lc] <= 0.0175:
                    logical_test[lc] = 1
            logical_test = np.mean(logical_test)
            if logical_test == 1 and j == 1:
                # If lines intercept at the same angle remove it
                to_remove.append(k)
            elif delta_x == 0 and delta_y == 0:
                # Remove lines that are dots
                # this is not possible due to the connectivity matrix but left here as an extra measure
                to_remove.append(k)
            else:
                image_array.append(image.img_to_array(j_image))
                # Add the node to the recorder
                node_recorder.append(i)
                # Add the maximum force to the force vector
                joint_force.append(np.max(d_force))

            # Close all the plots
            plt.close('all')
        else:
            to_remove.append(k)

    # Remove the fake joints from the array of node_lines
    if to_remove:
        node_lines = np.delete(node_lines, obj=to_remove, axis=0)

    # Indicate the availability of joints
    if len(node_recorder) > 0:
        joint_avail = True

        # Perform the prediction if there are joints
        node_recorder = perform_prediction(node_recorder, image_array)

        # Get the maximum strain energy on the joints
        fem_se = np.zeros([len(node_recorder), 1])
        i = 0
        for nj in range(len(node_recorder)):
            # Get the lines on this joint
            joint_node = int(node_recorder[nj, 0])
            this_joint = node_lines[nj, :]
            active_lines = np.array([i for i in this_joint if i != -1], dtype=int)
            line_se2 = np.zeros([len(active_lines) + 1, 1])
            if len(active_lines) == 2:
                j_mass.append(0.00008)
            elif len(active_lines) == 3:
                j_mass.append(0.00011)
            elif len(active_lines) == 3:
                j_mass.append(0.00015)
            elif len(active_lines) == 4:
                j_mass.append(0.00018)
            else:
                j_mass.append(0.00019)
            for i in range(len(active_lines)):
                # Get the nodes to focus on
                focus_node = line_nodes[int(active_lines[i]), :]
                for j in range(len(focus_node)):
                    if joint_node != focus_node[j]:
                        other_node = focus_node[j]
                        line_se2[i] = s_e[other_node]
            line_se2[i + 1] = s_e[joint_node]
            fem_se[nj] = np.max(line_se2)

        # un-normalise the predictions
        with open('CNN_Max_Min_Values.txt', 'rb') as filename:
            max_min_values = pickle.load(filename)
        max_v = max_min_values['maxSE']
        min_v = max_min_values['minSE']
        node_recorder[:, 1] = (((node_recorder[:, 1] - 1) / 9) * (max_v - min_v)) + min_v

        # Scale to 3D
        node_recorder[:, 1] = np.abs(node_recorder[:, 1])
        node_recorder[:, 1] = (1.851 * np.log(node_recorder[:, 1])) + 10.184
        node_recorder[:, 1] = np.abs(np.real(node_recorder[:, 1]))
        j_mass = np.array(j_mass)
        j_mass = (46936 * np.power(j_mass, 2)) + (45.391 * j_mass) - 0.0004
        j_mass = np.sum(j_mass)

        # Get the difference between the predicted and actual strain energies
        diff_se = np.zeros([len(node_recorder), 1])
        percent_se = np.zeros([len(node_recorder), 1])
        for i in range(len(node_recorder)):
            diff_se[i] = node_recorder[i, 1] - fem_se[i]
            percent_se[i] = fem_se[i] / node_recorder[i, 1]
            if percent_se[i] > 1:
                percent_se[i] = 1

        # Fail the structure is it has a failed joint
        diff_se = np.min(diff_se)
        if np.isnan(diff_se) or diff_se < 0:
            is_good = False

    return node_recorder, joint_avail, is_good, percent_se, j_mass


def perform_prediction(n_recorder, i_array):
    # Create a matrix to save the prediction and joint location
    output = np.zeros([len(n_recorder), 2])
    # Save the location in the output
    output[:, 0] = n_recorder

    # Load the CNN
    f = Path('cnnSE_model_structure.json')
    model_se_structure = f.read_text()
    model_se = model_from_json(model_se_structure)
    model_se.load_weights('cnnSE_model_weights.h5')

    # Perform prediction
    i_array = np.array(i_array)
    results_se = model_se.predict(i_array)
    output[:, 1] = np.array(results_se[:, 0])

    return output
