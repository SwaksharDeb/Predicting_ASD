import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
import pandas as pd
import glob

def calculate_color(velocity, acceleration, jerk):
    min_velocity, max_velocity = min(velocity), max(velocity)
    min_acceleration, max_acceleration = min(acceleration), max(acceleration)
    min_jerk, max_jerk = min(jerk), max(jerk)
    # Normalize velocity, acceleration, and jerk to the range [0, 1]
    vel_norm = (velocity - min_velocity) / (max_velocity - min_velocity)
    accel_norm = (acceleration - min_acceleration) / (max_acceleration - min_acceleration)
    jerk_norm = (jerk - min_jerk) / (max_jerk - min_jerk)

    # Calculate RGB values based on normalized values
    red = (vel_norm * 255)
    green =(accel_norm * 255)
    blue = (jerk_norm * 255)

    return red, green, blue


def interpolate_scanpath_data(time_points, positions_x, positions_y, num_points=5000):
    """
    Interpolate velocities, accelerations, and jerks for a given scanpath using linear interpolation.

    Parameters:
    - time_points: Time points corresponding to the scanpath data.
    - positions_x: X-coordinates of the scanpath.
    - positions_y: Y-coordinates of the scanpath.
    - num_points: Number of points for interpolation.

    Returns:
    - Interpolated time points, interpolated positions_x, interpolated positions_y,
      interpolated velocities, interpolated accelerations, and interpolated jerks.
    """
    # Use linear interpolation for positions
    interpolated_time = np.linspace(time_points[0], time_points[-1], num_points)
    interpolated_positions_x = np.interp(interpolated_time, time_points, positions_x)
    interpolated_positions_x = np.where(interpolated_positions_x >= 1, 0.9, interpolated_positions_x)  # Bounded position
    interpolated_positions_y = np.interp(interpolated_time, time_points, positions_y)
    interpolated_positions_y = np.where(interpolated_positions_y >= 1, 0.9, interpolated_positions_y)  # Bounded position

    # Calculate velocities, accelerations, and jerks for interpolated data
    interpolated_velocities = np.gradient(interpolated_positions_x, interpolated_time)
    interpolated_accelerations = np.gradient(interpolated_velocities, interpolated_time)
    interpolated_jerks = np.gradient(interpolated_accelerations, interpolated_time)

    return (
        interpolated_time,
        interpolated_positions_x,
        interpolated_positions_y,
        interpolated_velocities,
        interpolated_accelerations,
        interpolated_jerks
    )


# def interpolate_scanpath(scanpath_data, num_points=100):
#     """
#     Interpolate lines between consecutive scanpath points.

#     Parameters:
#     - scanpath_data: A list of tuples (x, y) representing gaze points over time.
#     - num_points: Number of interpolated points between each consecutive pair.

#     Returns:
#     - Interpolated scanpath data as a list of tuples (x, y).
#     """
#     interpolated_scanpath = []

#     for i in range(len(scanpath_data) - 1):
#         x1, y1 = scanpath_data[i]
#         x2, y2 = scanpath_data[i + 1]
#         interpolated_x = np.linspace(x1, x2, num_points, endpoint=False)
#         interpolated_y = np.linspace(y1, y2, num_points, endpoint=False)
#         interpolated_scanpath.extend(list(zip(interpolated_x, interpolated_y)))

#     # Add the last point of the original scanpath
#     interpolated_scanpath.append(scanpath_data[-1])

#     return interpolated_scanpath

def convert_scanpath_to_colored_image(scanpath_x, scanpath_y, image, interpolated_velocities, interpolated_accelerations, interpolated_jerks, line_width=2):
    """
    Convert eye scanpath data to an image-based visual form with color gradients based on movement dynamics.

    Parameters:
    - scanpath_data: A list of tuples (x, y) representing gaze points over time.
    - image_size: Tuple specifying the size of the output image.
    - line_width: Width of the lines representing the transitions.

    Returns:
    - A NumPy array representing the visual form of the scanpath.
    """
    # Create a blank image
    #image = np.zeros((*image_size, 3), dtype=np.uint8)

    # Interpolate scanpath lines
    #interpolated_scanpath = interpolate_scanpath(scanpath_data)

    # Normalize gaze points to image size
    #normalized_scanpath = [(int(x * image_size[0]), int(y * image_size[1])) for x, y in scanpath_data]


    #raf = np.array(interpolate_scanpath)
    # Create LineCollection for drawing lines with varying colors
    #segments = np.array(list(zip(normalized_scanpath, normalized_scanpath[1:])))
    #velocities = np.linalg.norm(np.diff(segments, axis=1), axis=2)
        
    # Normalize velocities to [0, 1]
    #normalized_velocities = (velocities - velocities.min()) / (velocities.max() - velocities.min()) *100

    # Assign colors based on velocity
    #colors = plt.cm.Reds(normalized_velocities)
    red, green, blue = calculate_color(interpolated_velocities, interpolated_accelerations, interpolated_jerks)
    #arr = np.array(normalized_scanpath[:-1])
    row = (scanpath_x*image_size[0]).astype(int)
    col = (scanpath_y*image_size[1]).astype(int)
    color_channel = np.concatenate((red.reshape(-1,1), green.reshape(-1,1),blue.reshape(-1,1)),-1)
    image[row, col] = color_channel
    
    return image

# Example usage:
#file = pd.read_csv ('demo/eye tracking/Jahanara_fixations.csv')
imgDir = 'demo/eye tracking/*.csv'
label_file = pd.read_csv ('demo/label.csv')
names = label_file.iloc[:,2].to_numpy()
names_list = list(names)
ground_truth = label_file.iloc[:,7].to_numpy()
for enum, imgFile in enumerate(glob.iglob(imgDir, recursive=True)):
    file = pd.read_csv (imgFile)
    patient_name = imgFile[18:-14]
    for y in range(0,len(names_list)):
        if str(names_list[y]) == 'nan':
            continue
        elif all(x in names_list[y].split() for x in patient_name.split()):
            phq_score = ground_truth[y]
            break
    times = file.iloc[:,3].to_numpy()
    fixations = file.iloc[:,5:7].to_numpy()
    sacads = file.iloc[:,-4:-2].to_numpy()
    times_list = []
    fixations_list = []
    sacads_list = []
    count_init = 0
    for i in range(0,len(times)):
        if i != len(times)-1 and times[i] > times[i+1]:
            #count += 1
        #else:
            count_end = i
            times_list.append(times[count_init:count_end])
            fixations_list.append(fixations[count_init:count_end])
            sacads_list.append(sacads[count_init:count_end])
            count_init = i+1
        
    #time_points = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    #scanpath_data = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.5], [0.6, 0.2], [0.3, 0.1], [0.2, 0.2], [0.9, 0.5], [0.6, 0.7]])
    for i in range(0,len(times_list)):
        fixation_data = fixations_list[i]
        sacads_data = sacads_list[i]
        fix_pos_x = np.abs(fixation_data[:, 0])
        fix_pos_y = np.abs(fixation_data[:, 1])
        #sacads_pos_x = sacads_data[:, 0]
        #sacads_pos_x = np.abs(sacads_pos_x / max(sacads_pos_x))
        #sacads_pos_y = sacads_data[:, 1]
        #sacads_pos_y = np.abs(sacads_pos_y / max(sacads_pos_y))
        time_points = times_list[i]
        
        if len(fixation_data)<4:
            continue
        # Interpolate data
        (
            interpolated_time,
            interpolated_fix_pos_x,
            interpolated_fix_pos_y,
            interpolated_fix_velocities,
            interpolated_fix_accelerations,
            interpolated_fix_jerks
        ) = interpolate_scanpath_data(time_points, fix_pos_x, fix_pos_y)
        
        
        # (
        #     _,
        #     interpolated_sacads_pos_x,
        #     interpolated_sacads_pos_y,
        #     interpolated_sacads_velocities,
        #     interpolated_sacads_accelerations,
        #     interpolated_sacads_jerks
        # ) = interpolate_scanpath_data(time_points, sacads_pos_x, sacads_pos_y)
        
        if i % 2 == 0: # take consecutive five images     
            image_size=(480, 640)
            image = np.zeros((*image_size, 3), dtype=np.uint8)
        
        image = convert_scanpath_to_colored_image(interpolated_fix_pos_x, interpolated_fix_pos_y, image, 
                                                        interpolated_fix_velocities, interpolated_fix_accelerations,
                                                        interpolated_fix_jerks)
        
        # image = convert_scanpath_to_colored_image(interpolated_sacads_pos_x, interpolated_sacads_pos_y, image,
        #                                                 interpolated_sacads_velocities, interpolated_sacads_accelerations,
        #                                                 interpolated_sacads_jerks)
        
        # Display the resulting image
        #plt.imshow(image_array[:, :, 0], cmap='gray')
        #plt.show()
        
        if i % 2 == 1: # save the last image 
            from PIL import Image
            img = Image.fromarray(image)
            if phq_score >= 15: 
                img.save('demo/pos/'+patient_name+'_'+str(i)+'.png')
            else:
                img.save('demo/neg/'+patient_name+'_'+str(i)+'.png')   
    