import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skimage
import pkg_resources

def load_distances(i: int = 0) -> pd.DataFrame:
    """
    Load distance data from a CSV file for a specific experiment.

    This function reads a CSV file containing distance data for a particular experiment and
    returns the data as a pandas DataFrame.

    Parameters:
    - i (int, optional): An integer specifying the experiment number. Default is 0.

    Returns:
    - distances (pd.DataFrame): A pandas DataFrame containing the loaded distance data expressed in meters and
      the timestamps at which the distances were measured expressed in nanoseconds.

    Example:
    To load distance data for experiment 0:
    >>> distances = load_distances(0)
    >>> print(distances.head())

    Output:
                    timestamp  distance
    0  1686733023749771520    2.5838
    1  1686733023783322624    2.5838
    2  1686733023816582912    2.5838
    3  1686733023849853952    2.5838
    4  1686733023883287808    2.5838

    Note:
    - The CSV file is expected to have columns 'timestamp' (nanoseconds) and 'distance' (meters).

    - The function assumes that the CSV file is located in a 'data' directory and follows
      a specific naming convention ('experiment_i/trackeri.csv').
    """
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #csv_file_path = os.path.join(script_dir, 'data', f'experiment_{i}', f'tracker{i}.csv')
    stream = pkg_resources.resource_stream(__name__, f"data/experiment_{i}/tracker{i}.csv")
    #return pd.read_csv(csv_file_path)
    return pd.read_csv(stream)

def load_range(i: int = 0) -> pd.DataFrame:
    """
    Load range data from a CSV file for a specific experiment and calculate the minimum values.

    This function reads a CSV file containing range data for a particular experiment and
    returns the data as a pandas DataFrame. It also calculates the minimum values along each row
    and appends them as a new 'min' column in the DataFrame. 
    Minimum values are calculated since range sensors during the experiments were placed with different
    angles with respect to the approching object and were able to detect the object in different moments.
    The minimum value is the minimum distance detected by the four sensors, thus it is the most conservative
    and safest value to use.

    Parameters:
    - i (int, optional): An integer specifying the experiment number. Default is 0.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame containing the loaded range data and the calculated
      minimum values.

    Example:
    To load range data for experiment 0 and display the first few rows:
    >>> range_data = load_range(0)
    >>> print(range_data.head())

    Output:
                timestamp  range_0  range_1  range_2  range_3    min
    0  1686733023805156096    2.136    5.633    4.516    5.523  2.136
    1  1686733023929177856    2.255    5.764    4.568    5.517  2.255
    2  1686733024029468160    2.215    5.670    4.567    5.444  2.215
    3  1686733024125935616    2.251    5.721    4.534    5.541  2.251
    4  1686733024231188224    2.271    5.710    4.512    5.594  2.271
    ...

    Notes:
    - The CSV file is expected to have columns 'timestamp' (nanoseconds) and 
      columns representing readings from each sensor, e.g., 'range_0', 'range_1', 'range_2', 'range_3' (meters).

    - The function assumes that the CSV file is located in a 'data' directory and follows
      a specific naming convention ('experiment_i/range_i.csv').

    - The 'min' column contains the minimum values of the range data for each row (meters).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'data', f'experiment_{i}', f'range_{i}.csv')
    df = pd.read_csv(csv_file_path)
    df['min'] = df.min(axis=1)
    return df

def load_images(experiment: int = 0, camera: str = "wheelchair", num_frame: int = 0):
    """
    Load an image frame from a specific experiment and camera.

    This function reads an image file from a specified experiment, camera, and frame number
    and returns the image as a NumPy array using OpenCV.

    Parameters:
    - experiment (int, optional): An integer specifying the experiment number. Default is 0.
    - camera (str, optional): A string specifying the camera source. Default is "wheelchair".
    - num_frame (int, optional): An integer specifying the frame number. Default is 0.

    Returns:
    - image (numpy.ndarray): An image frame represented as a NumPy array.

    Raises:
    - FileNotFoundError: If the image file for the specified experiment, camera, and frame
      number does not exist.

    Example:
    To load an image from experiment 0, camera "wheelchair", and frame number 0:
    >>> image = load_images(experiment=0, camera="wheelchair", num_frame=0)
    >>> cv2.imshow("Loaded Image", image)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()

    Notes:
    - The image file is expected to be in PNG format and follow a specific naming convention
      ('experiment_i/camera/frame_000i.png').

    - OpenCV (cv2) is used to read and handle the image data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.join(script_dir, 'data', f'experiment_{experiment}', camera, f'frame_{num_frame:04d}.png')
    if os.path.exists(img_file_path):
        image = cv2.imread(img_file_path)
        return image
    else:
        raise FileNotFoundError(f"Image file not found: {img_file_path}")
    
def load_and_plot_image(experiment: int = 0, camera: str = "wheelchair", num_frame: int = 0):
    """
    Load and plot an image frame from a specific experiment and camera.

    This function reads an image file from a specified experiment, camera, and frame number,
    displays it using matplotlib, and returns the image as a NumPy array using OpenCV.

    Parameters:
    - experiment (int, optional): An integer specifying the experiment number. Default is 0.
    - camera (str, optional): A string specifying the camera source. Default is "wheelchair".
    - num_frame (int, optional): An integer specifying the frame number. Default is 0.

    Returns:
    - image (numpy.ndarray): An image frame represented as a NumPy array.

    Raises:
    - FileNotFoundError: If the image file for the specified experiment, camera, and frame
      number does not exist.

    Example:
    To load and plot an image from experiment 0, camera "wheelchair", and frame number 0:
    >>> image = load_and_plot_image(experiment=0, camera="wheelchair", num_frame=0)

    Notes:
    - The image file is expected to be in PNG format and follow a specific naming convention
      ('experiment_i/camera/frame_000i.png').

    - OpenCV (cv2) is used to read and handle the image data, and matplotlib (plt) is used to
      display the image.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.join(script_dir, 'data', f'experiment_{experiment}', camera, f'frame_{num_frame:04d}.png')
    
    if os.path.exists(img_file_path):
        image = cv2.imread(img_file_path)
        

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        raise FileNotFoundError(f"Image file not found: {img_file_path}")
    
def load_full_experiment_images(experiment: int = 0, camera: str = "wheelchair"):
    """
    Load and return a list of image frames from a specific experiment and camera.

    This function reads all image files from a specified experiment and camera, sorts them in
    ascending order based on file names, and returns a list of image frames as NumPy arrays
    using OpenCV.

    Parameters:
    - experiment (int, optional): An integer specifying the experiment number. Default is 0.
    - camera (str, optional): A string specifying the camera source. Default is "wheelchair".

    Returns:
    - images (list of numpy.ndarray): A list of image frames, where each frame is represented as
      a NumPy array.

    Example:
    To load all image frames from experiment 0, camera "wheelchair":
    >>> images = load_full_experiment_images(experiment=0, camera="wheelchair")

    Notes:
    - The image files are expected to be in PNG format and follow a specific naming convention
      ('experiment_i/camera/frame_000i.png').

    - OpenCV (cv2) is used to read and handle the image data.

    - The list 'images' contains all image frames in ascending order of file names.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir_path = os.path.join(script_dir, 'data', f'experiment_{experiment}', camera)
    img_files = os.listdir(img_dir_path)
    img_files.sort()
    images = []
    for img_file in img_files:
        img_file_path = os.path.join(img_dir_path, img_file)
        image = cv2.imread(img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

def load_full_experiment_images_and_plot(experiment: int = 0, camera: str = "wheelchair", speed: float = 1):
      """
      Load and plot a sequence of images from a specific experiment and camera as an animation.

      This function loads a sequence of images from a specified experiment and camera and displays
      them as an animation using Matplotlib's FuncAnimation. You can control the animation speed
      with the 'speed' parameter.

      Parameters:
      - experiment (int, optional): An integer specifying the experiment number. Default is 0.
      - camera (str, optional): A string specifying the camera source. Default is "wheelchair".
      - speed (float, optional): A float specifying the animation speed. A value between 0.1 and 10
        controls the speed. A higher value makes the animation faster. Default is 1.

      Example:
      To load and animate images from experiment 0, camera "wheelchair" at double speed:
      >>> load_full_experiment_images_and_plot(experiment=0, camera="wheelchair", speed=2)

      Notes:
      - The function internally uses 'load_full_experiment_images' to load the image sequence.
      - Images are displayed as an animation, and you can control the speed with the 'speed' parameter.
      - The 'speed' parameter allows values between 0.1 (slow) and 10 (fast).
      """
      images = load_full_experiment_images(experiment, camera)

      # Create a figure and axis for the animation
      fig, ax = plt.subplots()
      im = ax.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))

      # Function to update the image in each frame
      def update(frame):
          im.set_array(images[frame])
          return im,

      # Create the animation
      if 0.1 < speed < 10:
        ani = FuncAnimation(fig, update, frames=len(images), blit=True,interval=33/speed)
      else:
        ani = FuncAnimation(fig, update, frames=len(images), blit=True,interval=33)

      # Show the animation (you can save it to a file with ani.save)
      plt.show()
