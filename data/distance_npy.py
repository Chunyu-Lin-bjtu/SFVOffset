import os
# import cv2
import numpy as np 
from colorsys import hsv_to_rgb
import copy

def create_distance_lidar(npy_path, save_dir):
    """ 生成npy文件并返回
    """
    print("%s is doing."%npy_path)
    
    (filepath, tempfilename) = os.path.split(npy_path)
    # (filename, extension) = os.path.splitext(tempfilename)
    record = np.load(npy_path)
    npydata_all = copy.deepcopy(record)
    npydata = npydata_all[:,:,0:3]
    npydata_i_d = npydata_all[:,:,3:6]

    tempdata = np.zeros([9,npydata.shape[0]+2,npydata.shape[1]+2,npydata.shape[2]])
    # print(tempdata.shape)
    tempdata_shape0 = tempdata.shape[0]
    tempdata_shape1 = tempdata.shape[1]
    tempdata_shape2 = tempdata.shape[2]
    tempdata[0,0:tempdata_shape1-2,0:tempdata_shape2-2,:] = npydata
    tempdata[1,1:tempdata_shape1-1,0:tempdata_shape2-2,:] = npydata
    tempdata[2,2:tempdata_shape1,0:tempdata_shape2-2,:] = npydata
    tempdata[3,0:tempdata_shape1-2,1:tempdata_shape2-1,:] = npydata
    tempdata[4,1:tempdata_shape1-1,1:tempdata_shape2-1,:] = npydata
    tempdata[5,2:tempdata_shape1,1:tempdata_shape2-1,:] = npydata
    tempdata[6,0:tempdata_shape1-2,2:tempdata_shape2,:] = npydata
    tempdata[7,1:tempdata_shape1-1,2:tempdata_shape2,:] = npydata
    tempdata[8,2:tempdata_shape1,2:tempdata_shape2,:] = npydata
    data_ = tempdata[0,:,:,:]+tempdata[1,:,:,:]+tempdata[2,:,:,:]+tempdata[3,:,:,:]+tempdata[5,:,:,:]+tempdata[6,:,:,:]+tempdata[7,:,:,:]+tempdata[8,:,:,:]-8*tempdata[4,:,:,:]
    # print(data_.shape)
    distance_data = data_[1:data_.shape[0]-1,1:data_.shape[1]-1,:]
    alldata = np.concatenate((distance_data,npydata_i_d),axis=2)
    # print(distance_data.shape)
    # print(alldata.shape)
    write_path = os.path.join(save_dir,tempfilename)
    np.save(write_path,alldata)

dir_path = './lidar_2d'
save_dir_path = './lidar_3dobj_dis'
txt_file = './train_usefile_3dobj.txt'

if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
with open(txt_file) as read_obj:
    npy_files = read_obj.readlines()
    for npy_file in npy_files:
        npy_file = npy_file.rstrip('\n')
        npy_full_name = os.path.join(dir_path,npy_file+'.npy')
        if not os.path.exists(npy_full_name):
            print("{:s}not exit.".format(npy_full_name))
            continue
        else:
            create_distance_lidar(npy_full_name,save_dir_path)
        
# path_list = os.listdir(dir_path)
# path_list.sort()
# for filename in path_list:
    # print(os.path.join(dir_path,filename))


