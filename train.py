# -*- coding: utf-8 -*-

#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn
import pickle
from gensim.models import Word2Vec

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

# Parameters
# ==================================================

# Data loading params

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("bl_data_file", "./data/two_cccs_lack_generalized_bl_over.txt", "Data source for the bl data.")
tf.flags.DEFINE_string("yg_data_file", "./data/two_cccs_lack_generalized_yg.txt", "Data source for the yg data.")
tf.flags.DEFINE_string("zz_data_file", "./data/two_cccs_lack_generalized_zz_over.txt", "Data source for the zz data.")
tf.flags.DEFINE_string("stop_word_file", "./data/stopword1208.txt", "Data source for stop words.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 141, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "Resume checkpoint")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# checkpoint_prefix global_variables
# tf.flags.DEFINE_string("checkpoint_prefix", "F:/Tensorflow/CNN_en_02/runs/checkpoints",
# "Comma-separated filter sizes (default: '3,4,5')")

FLAGS = tf.flags.FLAGS
# FLAGS.parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
# 意思是空一行
print("")

# Data Preparation
# ==================================================
# 生成对应于词典的词向量
relative_path = './relative_data/relative.txt'
relative_mark_path = './relative_data/relative_mark.xlsx'
wordfile = './data/word_embedding2.npy'
vocab_dic, reversed_vocab = data_helpers.file_to_vocab(FLAGS.bl_data_file, FLAGS.yg_data_file,
                                                       FLAGS.zz_data_file, FLAGS.stop_word_file)
data_helpers.file_to_wordembedding(FLAGS.bl_data_file, FLAGS.yg_data_file, FLAGS.zz_data_file)
Model = Word2Vec.load('./train_file/word_embedding2_stop.model')
data_helpers.word_to_embedding('./train_file/word_embedding2_stop.model', wordfile, vocab_dic, relative_path, relative_mark_path)

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data(FLAGS.bl_data_file, FLAGS.yg_data_file,
                                                          FLAGS.zz_data_file, FLAGS.stop_word_file)
# x, y = data_helpers.load_data_and_labels(FLAGS.bl_data_file, FLAGS.yg_data_file, FLAGS.zz_data_file)


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# 分离训练集和测试集，用交叉验证
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))

# 训练集和验证集
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# len(vocabulary)
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

print("Train data shape[1]: {:d}".format(x_train.shape[1]))
print("num_classes shape[1]: {:d}".format(y_train.shape[1]))

# Training
# ==================================================


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 实例化了TextCNN
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure 最优化网络的损失函数
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional) 梯度值和稀疏度
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries 对模型和概要的输出目录
        # 时间戳
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy 对损失和精度的概要
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries 训练概要
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory.Tensorflow assumes this directory already exists so we need to create it

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # pickle.dump(vocabulary, open("vocab", "wb"))
        # pickle.dump(vocabulary, open(os.path.join(out_dir, "vocab"), "wb"))

        # Initialize all variables
        # 也可以对特征的变量手动初始化，如训练好的词向量
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.initialize_all_variables())

        # 保存训练结果的参数
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint, 'checkpoints'))
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return loss

        bad_loss = float("inf")  # 全局变量
        patient = 5
        wait = 5
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            # 评估步骤  step 100, loss 0.261838, acc 0.929039
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                # 提前结束训练 early stop
                loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if loss < bad_loss:
                    print("-----判断之前，此时bad_loss={:g}-----".format(bad_loss))
                    bad_loss = loss
                    # wait = 0
                    # wait = wait + 1
                    print("-----判断之后，此时bad_loss={:g}-----".format(bad_loss))
                else:
                    # wait = wait + 1
                    if wait >= patient:
                        # if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        break
                # if loss < bad_loss and patient <= FLAGS.patient:
                #     bad_loss = loss
                #     patient = patient + 1
                # else:
                #     break
                print("")
