import gpxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geopy.distance import geodesic

# Read GPX file
def read_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    return gpx

# Convert GPX data to a Pandas DataFrame
def gpx_to_dataframe(gpx):
    data = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude
                })
                
    return pd.DataFrame(data)

# Calculate distance between two GPS coordinates (in meters)
def calculate_distance(point1, point2):
    return geodesic(point1, point2).meters

# Interpolation function to generate more points between two locations
def interpolate_points(lat1, lon1, lat2, lon2, num_points):
    latitudes = np.linspace(lat1, lat2, num_points)
    longitudes = np.linspace(lon1, lon2, num_points)
    return latitudes, longitudes

# Function to interpolate the whole route for smooth animation
def interpolate_route(df, total_frames):
    new_data = {
        'latitude': [],
        'longitude': []
    }
    
    total_distance = 0  # Track total distance
    
    # Calculate total distance of the entire route
    for i in range(1, len(df)):
        point1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        total_distance += calculate_distance(point1, point2)
    
    # Calculate how many frames are needed between each point
    distance_per_frame = total_distance / total_frames
    
    # Now interpolate between each pair of points
    for i in range(1, len(df)):
        point1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance = calculate_distance(point1, point2)
        
        # Determine how many frames this segment should take
        num_frames = max(1, int(distance / distance_per_frame))
        
        # Interpolate between points based on the number of frames
        latitudes, longitudes = interpolate_points(
            df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'], 
            df.iloc[i]['latitude'], df.iloc[i]['longitude'], 
            num_frames
        )
        
        # Add interpolated points to the new data
        new_data['latitude'].extend(latitudes)
        new_data['longitude'].extend(longitudes)
    
    # Convert to DataFrame
    df_interpolated = pd.DataFrame(new_data)
    return df_interpolated

# Function to animate the route
def animate_route(df, total_duration=30, fps=30):
    total_frames = total_duration * fps  # Total frames for 30 seconds at 30 fps
    
    # Interpolate the route to match the desired number of frames
    df_interpolated = interpolate_route(df, total_frames)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot static path
    ax.plot(df_interpolated['longitude'], df_interpolated['latitude'], color='gray', linestyle='-', marker=None, zorder=1)
    
    # Plot the original GPX points as larger black dots
    ax.scatter(df['longitude'], df['latitude'], color='black', s=50, zorder=3, label="Original GPX Points")
    
    # Create a scatter plot to animate the point
    point, = ax.plot([], [], 'ro', zorder=2)
    
    # Set axis labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Animated GPS Route ({total_duration} seconds)')
    
    # Set limits for the plot
    ax.set_xlim(df_interpolated['longitude'].min() - 0.001, df_interpolated['longitude'].max() + 0.001)
    ax.set_ylim(df_interpolated['latitude'].min() - 0.001, df_interpolated['latitude'].max() + 0.001)
    
    # Add legend to explain the black dots
    ax.legend(loc="upper left")
    
    # Initialization function for animation
    def init():
        point.set_data([], [])
        return point,

    # Animation function that updates the point position
    def update(frame):
        point.set_data([df_interpolated['longitude'].iloc[frame]], [df_interpolated['latitude'].iloc[frame]])
        return point,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(df_interpolated), init_func=init, interval=1000/fps, blit=True)

    plt.show()

    # Optionally, save the animation as a video file
    # ani.save('gps_route_animation.mp4', writer='ffmpeg', fps=fps)

# Specify the GPX file path
gpx_file_path = '0930051215-48030.gpx'

# Read the GPX file and convert to DataFrame
gpx = read_gpx(gpx_file_path)
df_gpx = gpx_to_dataframe(gpx)

# Animate the route with a fixed duration (e.g., 30 seconds)
animate_route(df_gpx, total_duration=10, fps=30)
