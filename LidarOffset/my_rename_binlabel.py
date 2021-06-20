import os
import numpy as np

root = os.getcwd()
raw_data = root+'/Kitti/2011_09_26'

def rename_bin(dir_path, file_name):
    for path in os.listdir(dir_path):
        print(path)
        if path[-4:] == '.bin':
            old_file = os.path.join(dir_path,path)
            new_file = os.path.join(dir_path,file_name[:11]+file_name[17:22]+path)
            os.rename(old_file,new_file)

def rename_label(dir_path, file_name):
    for path in os.listdir(dir_path):
        print(path)
        if path[-4:] == '.txt':
            old_file = os.path.join(dir_path,path)
            new_file = os.path.join(dir_path,file_name[:11]+file_name[17:22]+path)
            os.rename(old_file,new_file)



def list_dir_process(dir_path):
    for path in os.listdir(dir_path):
        print(path)
        if path[-4:] == 'sync':
            bin_path = raw_data + '/' + path + '/velodyne_points/data'
            label_path = raw_data + '/' + path + '/label'
            # print("Processing the ",raw_data + '/' + path)
            rename_bin(bin_path, path)
            rename_label(label_path, path)

list_dir_process(raw_data)