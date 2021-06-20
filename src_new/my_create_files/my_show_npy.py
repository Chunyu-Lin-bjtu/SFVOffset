""" show the *.npy file.
"""
import os
import cv2
import numpy as np 
from colorsys import hsv_to_rgb

# npy_path = "000000.npy"
npy_path = "000000.npy"
record = np.load(npy_path).astype(np.float32)
indentiy = record[:,:,3]
x_num = record[:,:,0]
y_num = record[:,:,1]
z_num = record[:,:,2]
print("x max:",np.max(x_num))
print("x min:",np.min(x_num))
print("y max:",np.max(y_num))
print("y min:",np.min(y_num))
print("z max:",np.max(z_num))
print("z min:",np.min(z_num))


# type_whitelist=['_background','car','van','truck','pedestrian','cyclist','person_sitting','tram','misc']
type_whitelist=['_background','car','pedestrian','cyclist']
n_instances = np.max(len(type_whitelist))
hues = np.linspace(0,1, n_instances+1)
np.random.shuffle(hues)
inst_colors = np.array([hsv_to_rgb(h, 0.7, 0.85) for h in hues])
inst_colors[0,:] = [0.4, 0.4, 0.4]

inst_colors[1,:] = [0.255     , 0.255     , 0.85      ]
inst_colors[2,:] = [0.255     , 0.65166667, 0.85      ]
inst_colors[3,:] = [0.65166667, 0.255     , 0.85      ]
inst_colors[4,:] = [0.85      , 0.255     , 0.65166667]


img_show = np.zeros([record.shape[0],record.shape[1],3])
for i_0 in range(record.shape[0]):
    for i_1 in range(record.shape[1]):
        # if record[i_0,i_1,3] > 0:
        temp_num = int(record[i_0,i_1,5])
        if temp_num > 0:
            inst_colors_ = np.array(inst_colors[temp_num,:])
        else :
            temp_indentiy = indentiy[i_0,i_1]
            if temp_indentiy < 0:
                temp_indentiy = 0
            inst_colors_ = np.array([temp_indentiy, temp_indentiy, temp_indentiy])
        # print(record[i_0,i_1,1])
        np.reshape(inst_colors_,(1,3))
        img_show[i_0,i_1,:] = inst_colors_

cv2.imshow("image",img_show)

cv2.waitKey()
