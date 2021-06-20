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

    v_res=26.9/64
    h_res=0.17578125

    with tf.Graph().as_default():
        mc = kitti_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc)

        saver = tf.train.Saver(model.model_params)
        counter = 0
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
                
                pc_min_num = pointCloud.min(0)
                pc_max_num = pointCloud.max(0)
                
                x_threshold = 0.5
                y_threshold = 0.5
                z_threshold = 0.5
                intensity_threshold = 0.3
                all_points_info = np.zeros([pointCloud.shape[0]],dtype=int)

                pc_x = pointCloud[:,0]
                pc_y = pointCloud[:,1]
                pc_z = pointCloud[:,2]
                pc_d = np.sqrt(pc_x ** 2 + pc_y ** 2 + pc_z ** 2)

                v_res_rad = np.radians(v_res)
                h_res_rad = np.radians(h_res)
                x_img_2 = np.arctan2(-pc_y, pc_x)
                x_img = np.floor((x_img_2 / h_res_rad)).astype(int)
                x_img -= np.min(x_img)
                    
                y_img_2 = -np.arcsin(pc_z/pc_d)
                y_img = np.round((y_img_2 / v_res_rad)).astype(int)
                y_img -= np.min(y_img)

                y_img[y_img >= 64] =63

                x_max = int(360.0 / h_res) + 1
                start_index = int(x_max/2 -256)
                x_img = x_img - start_index

                for i in range(pointCloud.shape[0]):
                    if x_img[i]<0 or x_img[i]>=512:
                        continue
                    if y_img[i]<0 or y_img[i]>63:
                        continue
                    temp_pred_cls = pred_cls[0, y_img[i], x_img[i]]
                    if pointCloud[i,2] < -1.5 :
                        continue
                    if pointCloud[i,0] > 10.0 :
                        continue
                    if temp_pred_cls !=0:
                        all_points_info[i] = temp_pred_cls

                from colorsys import hsv_to_rgb
                # close point cloud viewer if it is already open
                try:
                    viewer.close()
                except NameError:
                    pass

                viewer = pptk.viewer(pointCloud)
                print("pointCloud.shape: ", pointCloud.shape[0])
                viewer.set(point_size=0.01, show_axis=False, bg_color_bottom=[0.1,0.1,0.1,0.5])
                hues = np.linspace(0,1, mc.NUM_CLASS+1)
                # np.random.shuffle(hues)
                inst_colors = np.array([hsv_to_rgb(h, 0.7, 0.85) for h in hues])
                # print("inst_colors:",inst_colors)
                inst_colors[0,:] = [0.4, 0.4, 0.4]
                """
                for i in range(all_points_info.shape[0]):
                    if all_points_info[i]!=0:
                        print(all_points_info[i])
                """
                # inst_colors[0,:] = [0.5525, 0.85, 0.255]
                print(inst_colors[all_points_info,:].shape)
                viewer.attributes(inst_colors[all_points_info, :])



def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    detect()
    print('Detection output written to {}'.format(FLAGS.out_dir))


if __name__ == '__main__':
    tf.app.run()
