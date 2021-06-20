import numpy as np 
np_file = '000000.npy'

def lidar2d_offset(points, y_offset = 0.0, v_res=26.9/64, h_res=0.17578125, AZIMUTH_LEVEL=512, ZENITH_LEVEL=64):
    # , h_res=0.08
    r_lidar_ = points[:, :, 3]
    r_lidar_ = r_lidar_.ravel()

    x_lidar = points[:, :, 0]  # -71~73
    y_lidar = points[:, :, 1]  # -21~53
    z_lidar = points[:, :, 2]  # -5~2.6
    r_lidar = points[:, :, 3]  # Reflectance  0~0.99 
    d_lidar = points[:, :, 4]  # distance
    l_lidar = points[:, :, 5]  # label

    x_lidar = x_lidar.ravel()
    y_lidar = y_lidar.ravel()
    z_lidar = z_lidar.ravel()
    r_lidar = r_lidar.ravel()
    d_lidar = d_lidar.ravel()
    l_lidar = l_lidar.ravel()

    x_lidar = x_lidar[r_lidar_>=0]
    y_lidar = y_lidar[r_lidar_>=0]
    z_lidar = z_lidar[r_lidar_>=0]
    r_lidar = r_lidar[r_lidar_>=0]
    d_lidar = d_lidar[r_lidar_>=0]
    l_lidar = l_lidar[r_lidar_>=0]

    d_lidar_ = d_lidar
    x_lidar = x_lidar[d_lidar_>0]
    y_lidar = y_lidar[d_lidar_>0]
    z_lidar = z_lidar[d_lidar_>0]
    r_lidar = r_lidar[d_lidar_>0]
    d_lidar = d_lidar[d_lidar_>0]
    l_lidar = l_lidar[d_lidar_>0]



    y_lidar -= y_offset

    d = np.sqrt(x_lidar ** 2 + y_lidar ** 2 + z_lidar ** 2) + 1e-12

    # Convert res to Radians 
    v_res_rad = np.radians(v_res)
    h_res_rad = np.radians(h_res)

    # PROJECT INTO IMAGE COORDINATES
    # -1024~1024   -3.14~3.14  ;
    x_img_2 = np.arctan2(-y_lidar, x_lidar)
    # x_img_2 = -np.arcsin(y_lidar/r)  

    x_img = np.floor((x_img_2 / h_res_rad)).astype(int)  
    x_img -= np.min(x_img)  

    # -52~10  -0.4137~0.078
    # y_img_2 = -np.arctan2(z_lidar, r) #

    y_img_2 = -np.arcsin(z_lidar/d) 
    y_img = np.round((y_img_2 / v_res_rad)).astype(int)  
    print("y_img.min: ",np.min(y_img))
    print("y_img.max: ",np.max(y_img))
    y_img -= np.min(y_img) 
    # y_img[y_img >= 64] = 63 

    y_img[y_img >= 64] = 63 

    x_max = int(360.0 / h_res) + 1  
    # x_max = int(180.0 / h_res) + 1  

    depth_map = np.zeros((64, x_max, 6))#+255
    depth_map[y_img, x_img, 0] = x_lidar
    depth_map[y_img, x_img, 1] = y_lidar
    depth_map[y_img, x_img, 2] = z_lidar
    depth_map[y_img, x_img, 3] = r_lidar
    depth_map[y_img, x_img, 4] = d_lidar
    depth_map[y_img, x_img, 5] = l_lidar

    start_index = int(x_max/2 - 256)
    result = depth_map[:, start_index:(start_index+512), :]

    return result

record_npy = np.load(np_file).astype(np.float32) # x,y,z,intensity,d,label
record_npy_l = lidar2d_offset(record_npy, 0.0)
# record_npy_r = lidar2d_offset(record_npy, 0.0)
np.save("000000_l_05.npy",record_npy_l)
# np.save("000000_r_05.npy",record_npy_r)

