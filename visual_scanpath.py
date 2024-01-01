import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
import pandas as pd
import glob

def calculate_color(velocity, acceleration, jerk):
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
    Interpolate velocities, accelerations, and jerks for a given scanpath.

    Parameters:
    - time_points: Time points corresponding to the scanpath data.
    - positions_x: X-coordinates of the scanpath.
    - positions_y: Y-coordinates of the scanpath.
    - num_points: Number of points for interpolation.

    Returns:
    - Interpolated time points, interpolated positions_x, interpolated positions_y,
      interpolated velocities, interpolated accelerations, and interpolated jerks.
    """
    # Create a cubic spline for interpolation
    spline_x = CubicSpline(time_points, positions_x)
    spline_y = CubicSpline(time_points, positions_y)

    # Interpolate positions
    interpolated_time = np.linspace(time_points[0], time_points[-1], num_points)
    interpolated_positions_x = spline_x(interpolated_time)
    interpolated_positions_x = np.where(interpolated_positions_x>1,0.9,interpolated_positions_x)
    interpolated_positions_y = spline_y(interpolated_time)
    interpolated_positions_y = np.where(interpolated_positions_y>1,0.9,interpolated_positions_y)

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

def convert_scanpath_to_colored_image(scanpath_x, scanpath_y, image_size=(800, 600), line_width=2):
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
    image = np.zeros((*image_size, 3), dtype=np.uint8)

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
for i, imgFile in enumerate(glob.iglob(imgDir, recursive=True)):
    file = pd.read_csv (imgFile)
    patient_name = imgFile[18:-14]
    times = file.iloc[:,3].to_numpy()
    fixations = file.iloc[:,5:7].to_numpy()
    times_list = []
    fixations_list = []
    count_init = 0
    for i in range(0,len(times)):
        if i != len(times)-1 and times[i] > times[i+1]:
            #count += 1
        #else:
            count_end = i
            times_list.append(times[count_init:count_end])
            fixations_list.append(fixations[count_init:count_end])
            count_init = i+1
        
    #time_points = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    #scanpath_data = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.5], [0.6, 0.2], [0.3, 0.1], [0.2, 0.2], [0.9, 0.5], [0.6, 0.7]])
    for i in range(0,len(times_list)):
        scanpath_data = fixations_list[i]
        positions_x = scanpath_data[:, 0]
        positions_y = scanpath_data[:, 1]
        time_points = times_list[i]
        
        # Interpolate data
        (
            interpolated_time,
            interpolated_positions_x,
            interpolated_positions_y,
            interpolated_velocities,
            interpolated_accelerations,
            interpolated_jerks
        ) = interpolate_scanpath_data(time_points, positions_x, positions_y)
        
        # Normalize velocity, acceleration, and jerk for color mapping
        min_velocity, max_velocity = min(interpolated_velocities), max(interpolated_velocities)
        min_acceleration, max_acceleration = min(interpolated_accelerations), max(interpolated_accelerations)
        min_jerk, max_jerk = min(interpolated_jerks), max(interpolated_jerks)
        
        image_array = convert_scanpath_to_colored_image(interpolated_positions_x, interpolated_positions_y)
        
        # Display the resulting image
        #plt.imshow(image_array[:, :, 0], cmap='gray')
        #plt.show()
        
        from PIL import Image
        img = Image.fromarray(image_array)
        img.save('demo/images/'+patient_name+'_'+str(i)+'.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
