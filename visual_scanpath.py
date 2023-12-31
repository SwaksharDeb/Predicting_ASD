import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline


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


def interpolate_scanpath(scanpath_data, num_points=100):
    """
    Interpolate lines between consecutive scanpath points.

    Parameters:
    - scanpath_data: A list of tuples (x, y) representing gaze points over time.
    - num_points: Number of interpolated points between each consecutive pair.

    Returns:
    - Interpolated scanpath data as a list of tuples (x, y).
    """
    interpolated_scanpath = []

    for i in range(len(scanpath_data) - 1):
        x1, y1 = scanpath_data[i]
        x2, y2 = scanpath_data[i + 1]
        interpolated_x = np.linspace(x1, x2, num_points, endpoint=False)
        interpolated_y = np.linspace(y1, y2, num_points, endpoint=False)
        interpolated_scanpath.extend(list(zip(interpolated_x, interpolated_y)))

    # Add the last point of the original scanpath
    interpolated_scanpath.append(scanpath_data[-1])

    return interpolated_scanpath

def convert_scanpath_to_colored_image(scanpath_data, image_size=(800, 600), line_width=2):
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
    image = np.zeros((*image_size, 1), dtype=float)

    # Interpolate scanpath lines
    interpolated_scanpath = interpolate_scanpath(scanpath_data)

    # Normalize gaze points to image size
    normalized_scanpath = [(int(x * image_size[0]), int(y * image_size[1])) for x, y in interpolated_scanpath]


    raf = np.array(interpolate_scanpath)
    # Create LineCollection for drawing lines with varying colors
    segments = np.array(list(zip(normalized_scanpath, normalized_scanpath[1:])))
    velocities = np.linalg.norm(np.diff(segments, axis=1), axis=2)
        
    # Normalize velocities to [0, 1]
    normalized_velocities = (velocities - velocities.min()) / (velocities.max() - velocities.min()) *100

    # Assign colors based on velocity
    #colors = plt.cm.Reds(normalized_velocities)
    red, green, blue = calculate_color(velocities, accelerations, jerks)
    arr = np.array(normalized_scanpath[:-1])
    row = arr[:, 0]
    col = arr[:, 1]
    image[row, col] = normalized_velocities
    
    return image

# Example usage:
time_points = [1,2,3,4,5,6,7,8]
scanpath_data = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.5], [0.6, 0.2], [0.3, 0.1], [0.2, 0.2], [0.9, 0.5], [0.6, 0.7]])
positions_x = scanpath_data[:,0]
positions_y = scanpath_data[:,1]
velocities = np.gradient(positions_x, time_points)
accelerations = np.gradient(velocities, time_points)
jerks = np.gradient(accelerations, time_points)

# Normalize velocity, acceleration, and jerk for color mapping
min_velocity, max_velocity = min(velocities), max(velocities)
min_acceleration, max_acceleration = min(accelerations), max(accelerations)
min_jerk, max_jerk = min(jerks), max(jerks)


image_array = convert_scanpath_to_colored_image(scanpath_data)

# Display the resulting image
plt.imshow(image_array[:, :, 0], cmap='gray')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def interpolate_scanpath_data(time_points, positions_x, positions_y, num_points=100):
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
    interpolated_positions_y = spline_y(interpolated_time)

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

# Example usage:
time_points = np.array([1, 2, 3, 4, 5, 6, 7, 8])
scanpath_data = np.array([[0.2, 0.3], [0.4, 0.6], [0.8, 0.5], [0.6, 0.2], [0.3, 0.1], [0.2, 0.2], [0.9, 0.5], [0.6, 0.7]])
positions_x = scanpath_data[:, 0]
positions_y = scanpath_data[:, 1]

# Interpolate data
(
    interpolated_time,
    interpolated_positions_x,
    interpolated_positions_y,
    interpolated_velocities,
    interpolated_accelerations,
    interpolated_jerks
) = interpolate_scanpath_data(time_points, positions_x, positions_y)

# Now you can use the interpolated data as needed
