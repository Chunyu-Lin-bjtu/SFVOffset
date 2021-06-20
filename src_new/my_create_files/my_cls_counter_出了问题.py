# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from imdb import kitti
from utils.util import *
from nets import *
from skimage.morphology import closing,dilation
from skimage.measure import find_contours,label

import cv2
import linecache

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'val',
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/bichen/logs/squeezeSeg/eval',
                            """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/bichen/logs/squeezeSeg/train',
                            """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                             """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                             """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def eval_once(
    saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb,
    model):

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Restores from checkpoint
        saver.restore(sess, ckpt_path)
        # Assuming model_checkpoint_path looks something like:
        #   /ckpt_dir/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt_path.split('/')[-1].split('-')[-1]

        mc = model.mc
        mc.DATA_AUGMENTATION = False

        num_images = len(imdb.image_idx)

        _t = {
            'detect': Timer(),
            'read': Timer(),
            'eval': Timer()
        }

        tot_error_rate, tot_rmse, tot_th_correct = 0.0, 0.0, 0.0

        # class-level metrics
        tp_sum = np.zeros(mc.NUM_CLASS)
        fn_sum = np.zeros(mc.NUM_CLASS)
        fp_sum = np.zeros(mc.NUM_CLASS)
        # instance-level metrics
        itp_sum = np.zeros(mc.NUM_CLASS)
        ifn_sum = np.zeros(mc.NUM_CLASS)
        ifp_sum = np.zeros(mc.NUM_CLASS)
        # instance-level object matching metrics
        otp_sum = np.zeros(mc.NUM_CLASS)
        ofn_sum = np.zeros(mc.NUM_CLASS)
        ofp_sum = np.zeros(mc.NUM_CLASS)
        
        f_write = open('train_remove.txt','w')
        for i in xrange(int(num_images/mc.BATCH_SIZE)):
            file_idx_name = linecache.getline('val.txt', i+1).rstrip('\n')

            offset = max((i+1)*mc.BATCH_SIZE - num_images, 0)
            
            _t['read'].tic()
            lidar_per_batch, lidar_mask_per_batch, label_per_batch, _ \
                = imdb.read_batch(shuffle=False)
            _t['read'].toc()

            _t['detect'].tic()
            pred_cls = sess.run(
                model.pred_cls, 
                feed_dict={
                    model.lidar_input:lidar_per_batch,
                    model.keep_prob: 1.0,
                    model.lidar_mask:lidar_mask_per_batch
                }
            )
            # print(lidar_per_batch)
            lidar_per_batch_ = lidar_per_batch.copy()
            print(lidar_per_batch_.shape)
            pred_cls_ = np.squeeze(pred_cls) 
            print(pred_cls_.shape)
            pred_pedestrain = pred_cls_.copy()
            
            points_info = np.zeros([pred_cls_.shape[0],pred_cls_.shape[1],3])
            points_info[:,:,0:2] = lidar_per_batch_[0,:,:,0:2]
            points_info[:,:,2] = pred_cls_
            npy_write_path = file_idx_name + '.npy'
            npy_write_dir = 'remove_data'
            if os.path.exists(npy_write_dir)==False:
                os.makedirs(npy_write_dir)
            np.save(os.path.join(npy_write_dir,npy_write_path),points_info)
            # print("np.unique(pred_pedestrain):",np.unique(pred_pedestrain))
            # print("pred_pedestrain.shape",pred_pedestrain.shape)
            pred_pedestrain[pred_pedestrain!=2] = 0
            pred_pedestrain[pred_pedestrain==2] = 1
            pred_pedestrain = dilation(pred_pedestrain, selem=None, out=None)
            # pred_pedestrain = closing(pred_pedestrain, selem=None, out=None)

            # show the image
            img=np.zeros((pred_pedestrain.shape[0],pred_pedestrain.shape[1],3))
            img[:,:,0] = pred_pedestrain
            img[:,:,1] = pred_pedestrain
            img[:,:,2] = pred_pedestrain
            cv2.imshow("img",img)
            cv2.waitKey(10000)
            # print("np.unique(pred_pedestrain):",np.unique(pred_pedestrain))
            # print("sum:",sum(sum(pred_pedestrain)))

            pred_cyclist = pred_cls_
            pred_cyclist[pred_cyclist!=3] = 0
            pred_cyclist[pred_cyclist==3] = 1
            pred_cyclist = dilation(pred_cyclist, selem=None, out=None)
            # pred_cyclist = closing(pred_cyclist, selem=None, out=None)

            contours_pedestrain = find_contours(pred_pedestrain, 0.5)
            temp_contours_pedestrain = np.array(contours_pedestrain)
            # print(temp_contours_pedestrain.shape[0])
            contours_cyclist = find_contours(pred_cyclist, 0.5)

            list_pedestrain = []
            for i in range(len(contours_pedestrain)):
                contours_pedestrain_set = contours_pedestrain[i]
                min_contours_set_xindex = np.argmin(contours_pedestrain_set, axis=0)
                min_x, min_y = contours_pedestrain_set[min_contours_set_xindex[1]]
                # print("left:",min_x,min_y)
                max_contours_set_xindex = np.argmax(contours_pedestrain_set, axis=0)
                max_x, max_y = contours_pedestrain_set[max_contours_set_xindex[1]]
                # print("right:",max_x,max_y)
                x1 = lidar_per_batch[0, int(min_x), int(min_y), 1]
                x2 = lidar_per_batch[0, int(max_x), int(max_y), 1]
                x_distance = x1 - x2
                # print(x_distance)
                if x_distance>=0.1 and x_distance<=1.47:
                    list_pedestrain.append(x_distance)
                elif x_distance >= 1.47:
                    list_pedestrain.append(1.47)
            # print("list_pedestrain:",list_pedestrain)

            list_cyclist = []
            for i in range(len(contours_cyclist)):
                contours_cyclist_set = contours_cyclist[i]
                min_contours_set_xindex = np.argmin(contours_cyclist_set, axis=0)
                min_x, min_y = contours_cyclist_set[min_contours_set_xindex[1]]
                max_contours_set_xindex = np.argmax(contours_cyclist_set, axis=0)
                max_x, max_y = contours_cyclist_set[max_contours_set_xindex[1]]
                x1 = lidar_per_batch[0, int(min_x), int(min_y), 1]
                x2 = lidar_per_batch[0, int(max_x), int(max_y), 1]
                x_distance = x1 - x2
                if x_distance>=0.1 and x_distance<=2.34:
                    list_pedestrain.append(x_distance)
                elif x_distance >= 2.34:
                    list_pedestrain.append(2.34)
            # print("list_cyclist", list_cyclist)
            if len(list_pedestrain)==0:
                sum_pedestrain = 0
            else:
                sum_pedestrain = sum(list_pedestrain)
            # print("Average pedestrain:",average_pedestrain)
            if len(list_cyclist)==0:
                sum_cyclist = 0
            else:
                sum_cyclist = sum(list_cyclist) / len(list_cyclist)
            factor_pedestrain = 1
            factor_cyclist = 1
            if (len(list_cyclist)+len(list_pedestrain)) == 0:
                average_remove = 0.0
            else:
                average_remove = (sum_pedestrain+sum_cyclist)/(len(list_cyclist)+len(list_pedestrain))
            
            # write_string = str(file_idx_name) + ' ' + str(average_remove) + '\n'
            # print("write_str:",write_string)
            # f_write.write(write_string)

            # index_min = np.argmin(contours_pedestrain_set,)
            

            pred_cls_img = np.squeeze(pred_cls)
            # print("pred_cls_img.shape:",pred_cls_img.shape)
            pred_cls_img = closing(pred_cls_img, selem=None, out=None)
            pred_cls_counter = label(pred_cls_img,connectivity=2)
            # print(np.unique(pred_cls_counter))
            _t['detect'].toc()

            _t['eval'].tic()
            # Evaluation
            iou, tps, fps, fns = evaluate_iou(
                label_per_batch[:mc.BATCH_SIZE-offset],
                pred_cls[:mc.BATCH_SIZE-offset] \
                * np.squeeze(lidar_mask_per_batch[:mc.BATCH_SIZE-offset]),
                mc.NUM_CLASS
            )

            tp_sum += tps
            fn_sum += fns
            fp_sum += fps

            _t['eval'].toc()
            pred_ious = tps.astype(np.float)/(tps + fns + fps + mc.DENOM_EPSILON)
            print("pred_ious.shape:",pred_ious.shape)
            write_string = str(file_idx_name) + ' ' + str(average_remove) + '\n'
            # write_string = str(file_idx_name) + ' ' + str(pred_ious[0]) + ' ' + str(pred_ious[1]) +\
            #  ' ' + str(pred_ious[2]) + ' ' + str(pred_ious[3])+ '\n'
            print("write_str:",write_string)
            f_write.write(write_string)

            #   print ('detect: {:d}/{:d} im_read: {:.3f}s '
            #       'detect: {:.3f}s evaluation: {:.3f}s'.format(
            #             (i+1)*mc.BATCH_SIZE-offset, num_images,
            #             _t['read'].average_time/mc.BATCH_SIZE,
            #             _t['detect'].average_time/mc.BATCH_SIZE,
            #             _t['eval'].average_time/mc.BATCH_SIZE))

        ious = tp_sum.astype(np.float)/(tp_sum + fn_sum + fp_sum + mc.DENOM_EPSILON)
        pr = tp_sum.astype(np.float)/(tp_sum + fp_sum + mc.DENOM_EPSILON)
        re = tp_sum.astype(np.float)/(tp_sum + fn_sum + mc.DENOM_EPSILON)

        # print ('Evaluation summary:')
        # print ('  Timing:')
        # print ('    read: {:.3f}s detect: {:.3f}s'.format(
        #     _t['read'].average_time/mc.BATCH_SIZE,
        #     _t['detect'].average_time/mc.BATCH_SIZE
        # ))

        eval_sum_feed_dict = {
            eval_summary_phs['Timing/detect']:_t['detect'].average_time/mc.BATCH_SIZE,
            eval_summary_phs['Timing/read']:_t['read'].average_time/mc.BATCH_SIZE,
        }

        # print ('  Accuracy:')
        for i in range(1, mc.NUM_CLASS):
            print ('    {}:'.format(mc.CLASSES[i]))
            print ('\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}'.format(
                pr[i], re[i], ious[i]))
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_iou']] = ious[i]
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_precision']] = pr[i]
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/'+mc.CLASSES[i]+'_recall']] = re[i]

        eval_summary_str = sess.run(eval_summary_ops, feed_dict=eval_sum_feed_dict)
        for sum_str in eval_summary_str:
            summary_writer.add_summary(sum_str, global_step)
        summary_writer.flush()

        f_write.close()

def evaluate():
    """Evaluate."""
    assert FLAGS.dataset == 'KITTI', \
        'Currently only supports KITTI dataset'

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default() as g:

        assert FLAGS.net == 'squeezeSeg', \
            'Selected neural net architecture not supported: {}'.format(FLAGS.net)

        if FLAGS.net == 'squeezeSeg':
            mc = kitti_squeezeSeg_config()
            mc.LOAD_PRETRAINED_MODEL = False
            mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
            model = SqueezeSeg(mc)

        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

        eval_summary_ops = []
        eval_summary_phs = {}

        eval_summary_names = [
            'Timing/read', 
            'Timing/detect',
        ]
        for i in range(1, mc.NUM_CLASS):
            eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_iou')
            eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_precision')
            eval_summary_names.append('Pixel_seg_accuracy/'+mc.CLASSES[i]+'_recall')

        for sm in eval_summary_names:
            ph = tf.placeholder(tf.float32)
            eval_summary_phs[sm] = ph
            eval_summary_ops.append(tf.summary.scalar(sm, ph))

        saver = tf.train.Saver(model.model_params)

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        ckpts = set() 
        while True:
            if FLAGS.run_once:
                # When run_once is true, checkpoint_path should point to the exact
                # checkpoint file.
                eval_once(
                    saver, FLAGS.checkpoint_path, summary_writer, eval_summary_ops,
                    eval_summary_phs, imdb, model)
                return
            else:
                # When run_once is false, checkpoint_path should point to the directory
                # that stores checkpoint files.
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    if ckpt.model_checkpoint_path in ckpts:
                        # Do not evaluate on the same checkpoint
                        print ('Wait {:d}s for new checkpoints to be saved ... '
                                .format(FLAGS.eval_interval_secs))
                        time.sleep(FLAGS.eval_interval_secs)
                    else:
                        ckpts.add(ckpt.model_checkpoint_path)
                        print ('Evaluating {}...'.format(ckpt.model_checkpoint_path))
                        eval_once(
                            saver, ckpt.model_checkpoint_path, summary_writer,
                            eval_summary_ops, eval_summary_phs, imdb, model)
                else:
                    print('No checkpoint file found')
                    if not FLAGS.run_once:
                        print ('Wait {:d}s for new checkpoints to be saved ... '
                                .format(FLAGS.eval_interval_secs))
                        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        tf.gfile.MakeDirs(FLAGS.eval_dir)
        evaluate()


if __name__ == '__main__':
    tf.app.run()
