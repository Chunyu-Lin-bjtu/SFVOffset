import numpy as numpy


def load_kitti_lidar_data(filename, verbose=False, load_reflectance=False):
    """
    Loads lidar data stored in KITTI format.
    
    Parameters
    ----------
    filename
    verbose: Describes the number of point clouds loaded.

    Returns
    -------
    numpy.ndarray
        n_points by 4 array.
        Columns are x, y, z, reflectance

    """
    with open(filename, "rb") as lidar_file:
        # Velodyne data stored as binary float matrix
        lidar_data = np.fromfile(lidar_file, dtype=np.float32)
        # Velodyne data contains x,y,z, and reflectance
        lidar_data = lidar_data.reshape((-1,4))
    if verbose:
        print("Loaded lidar point cloud with %d points." % lidar_data.shape[0])
    if load_reflectance:
        return lidar_data
    else:
        return lidar_data[:,0:3]

velodyne_dir = "./dataset/KITTI/object/training/velodyne"
f_name = "000000"
velodyne_path = os.path.join(velodyne_dir,f_name+'.bin')
pointCloud = load_kitti_lidar_data(velodyne_path)

point_colors = np.zeros([pointCloud.shape[0],3], dtype=int)

for i in range(pointCloud.shape[0]):
    reflection = pointCloud[i,3]
    if reflection <= 0.5:
        point_colors[i, 0] = 1 - reflection * 2
        point_colors[i, 1] = reflection * 2
        point_colors[i, 2] = 0
    elif reflection > 0.5: 
        point_colors[i, 0] = 0
        point_colors[i, 1] = 1 - (reflection - 0.5) * 2
        point_colors[i, 2] = (reflection - 0.5) * 2

import pptk

viewer = pptk.viewer(pointCloud)
viewer.set(point_size=0.01, show_axis=False, bg_color_bottom=[0.1, 0.1, 0.1, 0.5])
viewer.attributes(point_colors)
