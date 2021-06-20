import os

root = os.getcwd()

raw_data = root+'/Kitti/2011_09_26'
type = 'label'
root_txt ='test.txt'
file_object = open(root_txt,'w')
a =0
raw_data_dir = os.listdir(raw_data)
raw_data_dir.sort()
for path in raw_data_dir:
    if path[-4:] == 'sync':
        Data_file = raw_data+'/'+path
        print(Data_file)
        Data_file_list = os.listdir(Data_file)
        Data_file_list.sort()
        for file in Data_file_list:
            if file ==type:
                lidar_data = Data_file + '/'+ file
                lidar_data_list = os.listdir(lidar_data)
                lidar_data_list.sort()
                for data in lidar_data_list:
                    print('data',data)
                    a= a+1
                    #print(a)
                    lidar_data_name = str(lidar_data[:-6])
                    data_name = str(data[:-4])
                    root_name = lidar_data_name +'/'+'velodyne_points/data/'+data_name+'.bin'
                    lidar_name = Data_file+'/'+'label/'+data_name+'.txt'
                    if os.path.exists(root_name) and os.path.exists(lidar_name):
                        print('root',root_name)
                        print('lidar',lidar_name)
                        file_object.write(root_name)
                        file_object.write(' ')
                        file_object.write(lidar_name)
                        file_object.write('\n')
                #file_object.close()