import numpy as np
import matplotlib.pyplot as plt


def poisson_process_on_road(n, length, width):
    # Generate n points with random x (length) and y (width) positions
    x_positions = np.random.uniform(0, length, n)
    y_positions = np.random.uniform(0, width, n)
    return x_positions, y_positions


def update_points(x_positions, y_positions, length, width, speed, dt):
    # Move points to the right (increase x positions)
    x_positions += speed * dt

    # Remove points that are out of bounds and add new ones
    valid_indices = x_positions <= length  # Keep points within the boundary
    x_positions = x_positions[valid_indices]
    y_positions = y_positions[valid_indices]

    # Add new points to maintain the count
    new_points_count = n - len(x_positions)
    new_x_positions = np.random.uniform(0, length, new_points_count)
    new_y_positions = np.random.uniform(0, width, new_points_count)

    # Combine the old valid points with the new ones
    x_positions = np.concatenate([x_positions, new_x_positions])
    y_positions = np.concatenate([y_positions, new_y_positions])

    return x_positions, y_positions


# Parameters
n = 10  # Number of points
length = 2000  # Length of the road in meters
width = 10  # Width of the road in meters
speed = 50  # Speed of movement (m/s)
dt = 0.1  # Time step for each update

# Generate initial points on the road
x_positions, y_positions = poisson_process_on_road(n, length, width)

# Create a figure
plt.figure(figsize=(12, 3))

# Simulation loop
for i in range(100):
    # Update the points' positions
    x_positions, y_positions = update_points(
        x_positions, y_positions, length, width, speed, dt
    )

    # Clear the plot and re-draw on the same figure
    plt.clf()  # Clear the figure instead of creating a new one
    plt.hlines(
        [0, width], 0, length, colors="gray", linestyles="solid"
    )  # Road boundaries
    plt.scatter(
        x_positions, y_positions, c="blue", s=100, label="Points"
    )  # Points on the road

    # Add labels and formatting
    plt.title(f"Spatial Poisson Process on Road: Iteration {i+1}")
    plt.xlim(0, length)
    plt.ylim(0, width)
    plt.xlabel("Position along the Road (m)")
    plt.ylabel("Width of the Road (m)")
    plt.xlim(0, length)
    plt.ylim(0 - 10, width + 10)

    # Display the updated plot without creating a new figure
    plt.pause(0.1)

# Show the final plot after all iterations
plt.show()
