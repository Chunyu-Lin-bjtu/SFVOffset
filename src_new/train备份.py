# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from imdb import kitti
from utils.util import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', './data/', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('train_dir', './mylog/train',
                            """Directory where to write event logs """
                            """and checkpoint.""") #tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeseg/train',"""Directory where to write event logs """"""and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', './data/SqueezeNet/squeezenet_v1.1.pkl',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 50,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def train():
  """Train SqueezeSeg model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default() as default_Graph:

    assert FLAGS.net == 'squeezeSeg', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    if FLAGS.net == 'squeezeSeg':
      mc = kitti_squeezeSeg_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeSeg(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    def enqueue(sess, coord):
      with coord.stop_on_exception():
        while not coord.should_stop():
          # read batch input
          lidar_per_batch, lidar_mask_per_batch, label_per_batch,\
              weight_per_batch = imdb.read_batch()

          feed_dict = {
              model.ph_keep_prob: mc.KEEP_PROB,
              model.ph_lidar_input: lidar_per_batch,
              model.ph_lidar_mask: lidar_mask_per_batch,
              model.ph_label: label_per_batch,
              model.ph_loss_weight: weight_per_batch,
          }

          sess.run(model.enqueue_op, feed_dict=feed_dict)

    #saver = tf.train.Saver(tf.all_variables())
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    #for v in tf.trainable_variables():
    for v in tf.global_variables():
      print(v.name,v.get_shape().as_list())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    conv1_kernels = sess.run(default_Graph.get_tensor_by_name('fire5/expand1x1/kernels:0'))
    conv1_kernels = np.array(conv1_kernels)
    print(conv1_kernels)
    print('conv1_kernels.max()',conv1_kernels.max())
    print('conv1_kernels.min()',conv1_kernels.min())
    # num = 1
    # assert num == 0
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    coord = tf.train.Coordinator()
    enq_threads = []
    for _ in range(mc.NUM_ENQUEUE_THREAD):
      eqth = threading.Thread(target=enqueue, args=[sess, coord])
      eqth.start()
      enq_threads.append(eqth)

    run_options = tf.RunOptions(timeout_in_ms=60000)

    try:
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps-1:
          kernel_output_ = sess.run(model.kernel_output['fire5/expand3x3'])
          op_list = [
              model.lidar_input, model.lidar_mask, model.label, model.train_op,
              model.loss, model.pred_cls, summary_op
          ]

          # lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
          #     _, loss_value, pred_cls, summary_str = sess.run(op_list,
          #                                                     options=run_options)
          output_prob_run = sess.run(model.output_prob, options=run_options)
          print('output_prob_run.max(): ',output_prob_run.max())
          print('output_prob_run.min(): ',output_prob_run.min())
          print('output_prob_run.shape: ',output_prob_run.shape)
          num = 0
          assert num == 1
          """
          print("///-----------------------------------///")
          conv_input_ = sess.run(model.inputs_thing)
          print("conv inputs:",conv_input_)
          print("conv inputs shape:",conv_input_.shape)
          print("///-----------------------------------///")
          print("conv inputs reduce_sum:",sess.run(tf.reduce_sum(model.inputs_thing)))
          """
          kernel_output_ = sess.run(model.kernel_output['fire5/expand3x3'])
          print("conv kernel_output:",kernel_output_)
          print("conv kernel_output shape:",kernel_output_.shape)
          # print("///-----------------------------------///")
          # print("conv1 kernel_output reduce_sum:",sess.run(tf.reduce_sum(model.kernel_output)))
          # biases_output_ = sess.run(model.biases_output)
          # print("conv biases_output:",biases_output_)
          # print("conv biases_output shape:",biases_output_.shape)
          # print("///-----------------------------------///")
          # print("conv1 biases_output reduce_sum:",sess.run(tf.reduce_sum(model.biases_output)))
          # conv_output_ = sess.run(model.conv_output)
          # print("conv outputs:",conv_output_)
          # print("conv outputs shape:",conv_output_.shape)
          # print("///-----------------------------------///")
          # print("conv1 inputs reduce_sum:",sess.run(tf.reduce_sum(model.conv_output)))
          
          """
          conv1_inputs_ = sess.run(model.conv1_input)
          print("conv1 inputs:",conv1_inputs_)
          print("conv1 inputs shape:",conv1_inputs_.shape)
          print("///-----------------------------------///")
          print("conv1 inputs reduce_sum:",sess.run(tf.reduce_sum(model.conv1_input)))
          conv1_outputs_ = sess.run(model.conv1_out)
          print("conv1 outputs:",conv1_outputs_)
          print("conv1 outputs shape:",conv1_outputs_.shape)
          print("///-----------------------------------///")
          print("conv1 reduce_sum:",sess.run(tf.reduce_sum(model.conv1_out)))
          """
          """
          print("lidar_input: ",'\n',sess.run(model.lidar_input))
          print("lidar_mask: ",'\n',sess.run(model.lidar_mask))
          print("label: ",'\n',sess.run(model.label))
          print("reduce_sum(model.label):",'\n',sess.run(tf.reduce_sum(model.label)))
          print("reduce_sum(model.lidar_mask: )",sess.run(tf.reduce_sum(model.lidar_mask)))
          outputlabel = sess.run(tf.reshape(model.label, (-1, )))
          label_sum = sess.run(tf.reduce_sum(model.label))
          print("sum(label):",label_sum)
          print("labels:",outputlabel)
          outputlogits = sess.run(tf.reshape(model.output_prob, (-1, mc.NUM_CLASS)))
          logits_sum = sess.run(tf.reduce_sum(model.output_prob))
          print("sum(outputlogits):",logits_sum)
          print("outputlogits:",outputlogits)
          print("///-----------------------------------///")
          fenzi = sess.run(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(model.label, (-1, )),logits=tf.reshape(model.output_prob, (-1, mc.NUM_CLASS))) * tf.reshape(model.lidar_mask, (-1, )) * tf.reshape(model.loss_weight, (-1, ))))
          print("fenzi: ",fenzi)
          print("-------is nan:-------",np.isnan(loss_value))
          print("-------loss_value-------",loss_value)
          print("------------label map------------")
          print(pred_cls[:6,:,:])
          """

          label_image = visualize_seg(label_per_batch[:6, :, :], mc)
          pred_image = visualize_seg(pred_cls[:6, :, :], mc)

          # Run evaluation on the batch
          ious, _, _, _ = evaluate_iou(
              label_per_batch, pred_cls*np.squeeze(lidar_mask_per_batch),
              mc.NUM_CLASS)

          feed_dict = {}
          # Assume that class-0 is the background class
          for i in range(1, mc.NUM_CLASS):
            feed_dict[model.iou_summary_placeholders[i]] = ious[i]

          iou_summary_list = sess.run(model.iou_summary_ops[1:], feed_dict)

          # Run visualization
          viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]
          viz_summary_list = sess.run(
              viz_op_list, 
              feed_dict={
                  model.depth_image_to_show: lidar_per_batch[:6, :, :, [4]],
                  model.label_to_show: label_image,
                  model.pred_image_to_show: pred_image,
              }
          )

          # Add summaries
          summary_writer.add_summary(summary_str, step)

          for sum_str in iou_summary_list:
            summary_writer.add_summary(sum_str, step)

          for viz_sum in viz_summary_list:
            summary_writer.add_summary(viz_sum, step)

          # force tensorflow to synchronise summaries
          summary_writer.flush()

        else:
          _, loss_value = sess.run(
              [model.train_op, model.loss], options=run_options)

        duration = time.time() - start_time
        print("loss_value is nan : ",np.isnan(loss_value))
        

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                    class_loss)

        if step % 10 == 0:
          num_images_per_step = mc.BATCH_SIZE
          images_per_sec = num_images_per_step / duration
          sec_per_batch = float(duration)
          format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               images_per_sec, sec_per_batch))
          sys.stdout.flush()

        # Save the model checkpoint periodically.
        if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps-1:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
    except Exception, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      sess.run(model.q.close(cancel_pending_enqueues=True))
      coord.join(enq_threads)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
