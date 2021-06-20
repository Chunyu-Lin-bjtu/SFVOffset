# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time
import glob    

import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image

from config import *
from imdb import kitti
from utils.util import *
from nets import *

import pptk

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'checkpoint', './data/SqueezeSeg_my/model.ckpt-24990',
        """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
        'input_path', './data/samples/*',
        """Input lidar scan to be detected. Can process glob input such as """
        """./data/samples/*.npy or single input.""")
tf.app.flags.DEFINE_string(
        'velodyne_dir', './data/velodyne',
        """Directory of the lidar points cloud.""")
tf.app.flags.DEFINE_string(
        'out_dir', './data/samples_out/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

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

def _normalize(x):
    return (x - x.min())/(x.max() - x.min())

def detect():
    """Detect LiDAR data."""

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default():
        mc = kitti_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc)

        saver = tf.train.Saver(model.model_params)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)
            for f in glob.iglob(FLAGS.input_path):
                (f_path, tempfname) = os.path.split(f)
                (f_name, f_extension) = os.path.splitext(tempfname)
                velodyne_path = os.path.join(FLAGS.velodyne_dir,f_name+'.bin')
                pointCloud = load_kitti_lidar_data(velodyne_path) # ------------------------------
                lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]
                lidar_mask = np.reshape(
                    (lidar[:, :, 4] > 0),
                    [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
                )
                lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD

                pred_cls = sess.run(
                    model.pred_cls,
                    feed_dict={
                        model.lidar_input:[lidar],
                        model.keep_prob: 1.0,
                        model.lidar_mask:[lidar_mask]
                    }
                )

                # save the data
                file_name = f.strip('.npy').split('/')[-1]
                np.save(
                    os.path.join(FLAGS.out_dir, 'pred_'+file_name+'.npy'),
                    pred_cls[0]
                )
                print("pred_cls.shape:", pred_cls.shape)
                # print("pred_cls.max:",np.max(pred_cls, axis=2))
                # print("pred_cls.min:",np.min(pred_cls, axis=2))

                # save the plot
                depth_map = Image.fromarray(
                    (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
                label_map = Image.fromarray(
                    (255 * visualize_seg(pred_cls, mc)[0]).astype(np.uint8))
                print("-----------label_map.shape:",label_map.size)

                blend_map = Image.blend(
                    depth_map.convert('RGBA'),
                    label_map.convert('RGBA'),
                    alpha=0.4
                )

                blend_map.save(
                    os.path.join(FLAGS.out_dir, 'plot_'+file_name+'.png'))
                
                """
                pc_min_num = pointCloud.min(0)
                pc_max_num = pointCloud.max(0)
                pc_x_range = pc_max_num[0] - pc_min_num[0]
                pc_y_range = pc_max_num[1] - pc_min_num[1]
                pc_z_range = pc_max_num[2] - pc_min_num[2]
                # print("x_range:",x_range,"\ny_range:",y_range,"\nz_range:",z_range)

                pc_x_box_len = 1
                pc_y_box_len = 1
                pc_z_box_len = 1
                voxel_list = np.zeros([pointCloud.shape[0]],dtype = int)
                # print(voxel_list.shape)
                for i in range(pointCloud.shape[0]):
                    voxel_num = int((pointCloud[i,0]-pc_min_num[0])//pc_x_box_len) + \
                        int((pointCloud[i,1]-pc_min_num[1])//pc_y_box_len*(int(pc_x_range//pc_x_box_len))) + \
                        int((pointCloud[i,2]-pc_min_num[2])//pc_z_box_len*(int(pc_x_range//pc_x_box_len))*(int(pc_y_range//pc_y_box_len)))
                    voxel_list[i] = voxel_num
                print("voxel_list.len:",len(voxel_list))
                print("pointCloud.shape:",pointCloud.shape[0])
                all_points_info = np.zeros([pointCloud.shape[0],], dtype=int)
                print("lidar.shape:",lidar.shape)
                for i_0 in range(lidar.shape[0]):
                    for i_1 in range(lidar.shape[1]):
                        temp_x = lidar[i_0, i_1, 0]
                        temp_y = lidar[i_0, i_1, 1]
                        temp_z = lidar[i_0, i_1, 2]
                        voxel_num = int((temp_x-pc_min_num[0])//pc_x_box_len+1) + \
                            int((temp_y-pc_min_num[1])//pc_y_box_len*(int(pc_x_range//pc_x_box_len))) + \
                            int((temp_z-pc_min_num[2])//pc_z_box_len*(int(pc_x_range//pc_x_box_len))*(int(pc_y_range//pc_y_box_len)))
                        voxel_index = np.argwhere(voxel_list==voxel_num)
                        for i_counter in range(voxel_index.shape[0]):
                            temp_pixel = pred_cls[0,i_0,i_1]
                            if temp_pixel!=0:
                                all_points_info[voxel_index[i_counter]] = temp_pixel
                """
                from colorsys import hsv_to_rgb
                # close point cloud viewer if it is already open
                try:
                    viewer.close()
                except NameError:
                    pass

                viewer = pptk.viewer(pointCloud)
                viewer.set(point_size=0.01, show_axis=False, bg_color_bottom=[0.1,0.1,0.1,0.5])
                hues = np.linspace(0,1, mc.NUM_CLASS+1)
                # np.random.shuffle(hues)
                inst_colors = np.array([hsv_to_rgb(h, 0.7, 0.85) for h in hues])
                # print("inst_colors:",inst_colors)
                inst_colors[0,:] = [0.4, 0.4, 0.4]
                for i in range(all_points_info.shape[0]):
                    if all_points_info[i]!=0:
                        print(all_points_info[i])
                # inst_colors[0,:] = [0.5525, 0.85, 0.255]
                viewer.attributes(inst_colors[all_points_info, :])



def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    detect()
    print('Detection output written to {}'.format(FLAGS.out_dir))


if __name__ == '__main__':
    tf.app.run()
