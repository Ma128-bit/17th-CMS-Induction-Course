import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import copy, math
import warnings, contextlib, io
warnings.filterwarnings("ignore", module="uhi")

def groupbylumi(df, lumimax, help=False):
    """
    Groups lumisections in the DataFrame based on a cumulative luminosity threshold.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a "mean_lumi" column.
        lumimax (float): Maximum allowed cumulative luminosity before starting a new group.

    Returns:
        list: A list of group IDs corresponding to each row in the input DataFrame.
    """
    if help:
        return groupbylumi.__doc__

    lumisum = 0
    groupID = 0
    groups = []
    run = df["run_number"][0]

    for i, lumi in enumerate(df["mean_lumi"]):
        if df["run_number"][i] != run:
            groupID += 1
            lumisum = 0
            run = df["run_number"][i]
            
        lumisum += lumi
        groups.append(groupID)

        if lumisum > lumimax:
            lumisum = 0
            groupID += 1

    return groups

def sum_imgs(group, help=False):
    """
    Aggregates multiple rows of a DataFrame into a single entry by summing up key values and images.

    Args:
        group (pandas.DataFrame): A DataFrame containing columns "mean_lumi", "entries",
                                  "run_number", "ls_number", and "data" (2D images).

    Returns:
        pandas.Series: A series containing the summed values and a merged image.
    """
    if help:
        return sum_imgs.__doc__
    
    return pd.Series({
        "lumi": group["mean_lumi"].sum(),
        "entries": group["entries"].sum(),
        "run_min": group["run_number"].iloc[0],
        "run_max": group["run_number"].iloc[-1],
        "ls_min": group["ls_number"].iloc[0],
        "ls_max": group["ls_number"].iloc[-1],
        "img": np.array([np.array(v) for v in np.sum(group["data"], axis=0)], dtype=np.float64),
    })

def Show2Dimg(img, title='CSC occupancy', help=False):
    """
    Displays a 2D image with CMS style using mplhep, flipping along the diagonal.

    Args:
        img (numpy.ndarray): 2D array representing the image.
        title (str, optional): Title of the plot. Defaults to 'CSC occupancy'.
    """
    if help:
        return Show2Dimg.__doc__

    # Set CMS style
    hep.style.use("ROOT")

    fig = plt.figure(figsize =(6, 5))
    img_temp = copy.deepcopy(img)
    
    max_ = np.max(img_temp)
    img_temp[img_temp == 0] = np.nan
    
    img_temp = img_temp.T 
    ybins = np.arange(img_temp.shape[0] + 1)
    xbins = np.arange(img_temp.shape[1] + 1)
    with contextlib.redirect_stdout(io.StringIO()):
        hep.hist2dplot(img_temp, xbins, ybins, cbarsize="5%", flow=False)
    
    plt.title(title, fontsize=18)  
    plt.tick_params(axis='both', labelsize=16) 
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    del img_temp
    
def mask_outside_radius(img, center, max_distance, help=False):
    """
    Sets to zero all values in the matrix img that are beyond max_distance from the center.
    
    Args:
        img (numpy.ndarray): 2D matrix.
        center (tuple): (x, y) coordinates of the center.
        max_distance (float): Maximum allowed radius.

    Returns:
        numpy.ndarray: Matrix with out-of-radius values set to 0.
    """
    if help:
        return Show2Dimg.__doc__
    
    x_center, y_center = center
    ny, nx = img.shape  # Matrix dimensions

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Compute the Euclidean distance from the center
    distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    # Mask values outside the radius
    masked_img = np.where(distance <= max_distance, img, 0)

    return masked_img

def Show2DLoss(img, vmin=-1.5, vmax=2., title='Loss', help=False):
    """
    Visualizes a 2D matrix (image) using a heatmap.
    
    Args:
        img (numpy.ndarray or torch.Tensor): 2D array or tensor to be visualized.
        vmin (float, optional): Minimum value for the color scale. Default is -1.5.
        vmax (float, optional): Maximum value for the color scale. Default is 2.0.
        title (str, optional): Title of the plot. Default is 'Loss'.

    Returns:
        None: The function displays the heatmap plot without returning any values.
    """
    if help:
        return Show2Dimg.__doc__

    hep.style.use("ROOT")

    fig = plt.figure(figsize =(6, 5))
    img_temp = copy.deepcopy(img)
    
    max_ = np.max(img_temp)
    img_temp[img_temp>vmax] = vmax
    img_temp[img_temp<vmin] = vmin
    img_temp[img_temp == 0] = np.nan
    
    img_temp = img_temp.T 
    ybins = np.arange(img_temp.shape[0] + 1)
    xbins = np.arange(img_temp.shape[1] + 1)
    with contextlib.redirect_stdout(io.StringIO()):
        hep.hist2dplot(img_temp, xbins, ybins, cbarsize="5%", cmin=vmin, cmax=vmax)
    
    plt.title(title, fontsize=18)  
    plt.tick_params(axis='both', labelsize=16) 
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    del img_temp


def rebin_image(image, binary_matrix, help=False):
    """
    This function takes an image and a binary matrix as input, applies radial binning
    to the image, and returns a re-binned image where each radial sector is averaged.
    The binary matrix is used to mask certain areas during the re-binning process.

    Parameters:
    - image (2D array): The input image to be processed.
    - binary_matrix (2D array): A binary matrix of the same shape as the image, used to mask certain areas.

    Returns:
    - arr3 (2D array): The re-binned image with mean values computed for each radial sector.
    """
    if help:
        return Show2Dimg.__doc__
    
    # Number of sectors
    num_sectors = 18 * 4
    angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)

    r, c = image.shape
    center = (49.5, 49.5)
    
    # Create polar coordinate grid
    y, x = np.indices((r, c))
    theta = np.arctan2(y - center[0], x - center[1])+np.pi/72  # Polar angle
    theta[theta < 0] += 2 * np.pi  # Ensure all angles are positive

    # Sector mapping
    rebinning = np.digitize(theta, angles)
    rebinning = rebinning * binary_matrix

    flat_binning = rebinning.flatten()
    flat_image = image.flatten()

    non_zero_indices = flat_binning != 0
    _, inverse_indices = np.unique(flat_binning[non_zero_indices], return_inverse=True)

    flat_image = np.where(np.isnan(flat_image), 0, flat_image)

    sum_vals = np.bincount(inverse_indices, weights=flat_image[non_zero_indices])

    count_vals = np.bincount(inverse_indices)

    mean_vals = sum_vals / count_vals

    arr3 = np.copy(image)

    arr3[rebinning != 0] = mean_vals[inverse_indices]

    return arr3    

def plot_LSs(monitoring_elements, run, ls, help=False):
    """
    Plots multiple 2D histograms for different Luminosity Sections (LS).
    
    Parameters:
    - `monitoring_elements`: DataFrame containing monitoring data.
    - `run`: Run number to filter the data.
    - `ls`: Tuple (ls_min, ls_max) defining the LS range.
    - `help`: If True, returns the documentation of Show2Dimg.
    """
    if help:
        return Show2Dimg.__doc__
    
    df_temp = monitoring_elements[
        (monitoring_elements["run_number"] == run) & 
        (monitoring_elements["ls_number"] > ls[0]) & 
        (monitoring_elements["ls_number"] < ls[1])
    ].reset_index()
    
    num_plots = len(df_temp)
    num_rows = math.ceil(num_plots / 4)

    fig, axes = plt.subplots(num_rows, 4, figsize=(16, num_rows * 4))
    axes = axes.flatten()  # Ensure axes is a 1D array for iteration

    hep.style.use("ROOT")

    for i, ax in enumerate(axes):
        if i < num_plots:
            plt.sca(ax)  # Set the current axis
            
            # Convert data column to numpy array
            img_temp = np.array([np.array(v) for v in df_temp["data"][i]], dtype=np.float64)
            img_temp[img_temp == 0] = np.nan  # Replace zeros with NaNs
            img_temp = img_temp.T  # Transpose to match expected orientation
            
            # Define bin edges
            ybins = np.arange(img_temp.shape[0] + 1)
            xbins = np.arange(img_temp.shape[1] + 1)
            
            # Suppress stdout output from hep.hist2dplot
            with contextlib.redirect_stdout(io.StringIO()):
                hep.hist2dplot(img_temp, xbins, ybins, cbarsize="5%", flow=False)

            ax.set_title(f"LS {ls[0] + i}")  # Set title for each LS plot
        ax.axis('off')  

    plt.tight_layout()  # Improve subplot spacing
    plt.show()
    
def show_img_reco_Loss(v1, v2, v3, id, vmin=-1.5, vmax=2.0, help=False):
    """
    Displays three side-by-side images: Img, Reco, and Loss.
    - `v1`, `v2`, `v3`: arrays containing the images.
    - `id`: index of the image to display.
    - `vmin`, `vmax`: limits for the Loss visualization.
    - `help`: if True, returns the documentation of Show2Dimg.
    """
    if help:
        return Show2Dimg.__doc__

    hep.style.use("ROOT")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    imgs_ = [v1[id], v2[id], v3[id]]
    titles = ["Img", "Reco", "Loss"]

    for i, ax in enumerate(axes.flatten()):
        plt.sca(ax)  # Set current axis
        
        img_temp = np.copy(imgs_[i])  # Create a copy to avoid modifying the original data
        
        if i == 2:  # Apply limits only for the Loss image
            img_temp = np.clip(img_temp, vmin, vmax)

        img_temp[img_temp == 0] = np.nan  # Set zero values to NaN for better visualization
        img_temp = img_temp.T  # Transpose to match expected orientation

        # Define bin edges
        ybins = np.arange(img_temp.shape[0] + 1)
        xbins = np.arange(img_temp.shape[1] + 1)

        # Suppress stdout output from hep.hist2dplot
        with contextlib.redirect_stdout(io.StringIO()):
            if i==2:
                hep.hist2dplot(img_temp, xbins, ybins, cbarsize="5%", cmin=vmin, cmax=vmax, flow=False)
            else:
                hep.hist2dplot(img_temp, xbins, ybins, cbarsize="5%", flow=False)

        ax.set_title(titles[i])  # Set subplot title
        ax.axis('off')
        
        del img_temp
    plt.tight_layout()
    plt.show()