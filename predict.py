# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import time
import datetime
import data_helpers
import csv
from text_cnn import TextCNN
import pickle
import json
import copy
from sklearn import metrics

# Parameters
# ==================================================
# 保存训练模型参数的路径,这个只是获取了当前的文件夹，如何调用最新训练好参数模型的文件
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

# Data loading params
# 调用flags内部的DEFINE_string函数来制定解析规则
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("bl_data_file", "./test data/two_cccs_lack_generalized_bl_test.txt", "Data source for the bl data.")
tf.flags.DEFINE_string("yg_data_file", "./test data/two_cccs_lack_generalized_yg_test_modify.txt", "Data source for the yg data.")
tf.flags.DEFINE_string("zz_data_file", "./test data/two_cccs_lack_generalized_zz_test.txt", "Data source for the zz data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1516342657/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("vocab_dir", "./data", "Vocabulary directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================
# 需要重新定义一个函数
# 测试集和训练集使用同一个vocabulary，所以不能用下列的方式得到测试输入x
# x, vocabulary, vocabulary_inv = data_helpers.load_data_process_predict(FLAGS.bl_data_file,
# FLAGS.yg_data_file, FLAGS.zz_data_file)
# print(x)

if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.bl_data_file, FLAGS.yg_data_file, FLAGS.zz_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["大企业来治小企业也可以，小企业来治大企业也可以。",
             "近日出现的雨雪天气，是由于受华北冷空气和华南高空暖湿气流的共同影响。",
             "但是我认为，小孩子智力上的差异主要取决于他们所受到的教育。"]
    y_test = [0, 1, 2]

# print(x_raw)
# print(y_test)   # 返回的是每一行中最大值的下标

# Map data into vocabulary

# vocab_path = os.path.join(FLAGS.vocab_dir, "", "vocab")
# max_document_length = max([len(str(x).split(",")) for x in x_raw])
# print(max_document_length)  # 68

# vocab_path = os.path.join(FLAGS.vocab_dir, "", "vocabulary.pickle")

# vocab_processor = learn.preprocessing.VocabularyProcessor(14)
# # vocab_processor.restore(vocab_path)
# x_raw = [' '.join(x) for x in x_raw]
# x_test = np.array(list(vocab_processor.transform(x_raw)))

# x_raw 是多维数组，将每一个list中词对应词典中的index,最终将结果保存到x_test

# vocab_path = "./data/vocabulary.pickle"
# reversed_vocab_path = "./data/reversed_vocabulary.pickle"
# # 词表存在则直接加载
# if os.path.exists(vocab_path) and os.path.exists(reversed_vocab_path):
#     with open(vocab_path, "rb") as file:
#             dict = pickle.load(file)

max_document_length = max([len(str(x).split(",")) for x in x_raw])
print(max_document_length)
# min_document_length = 14
# max_document_length = 88
x_raw_num = copy.deepcopy(x_raw)
for index_Big, word_list in enumerate(x_raw_num):
    for index, word in enumerate(word_list):
        # enumerate(word_list)
        with open("./data/vocab.json", 'r') as file:
            dict = json.load(file)
            word_list[index] = dict.get(word, 0)
    while len(word_list) < max_document_length:
        word_list.append(0)
    else:
        word_list = word_list[0:max_document_length]
    x_raw_num[index_Big] = word_list

# print(x_raw)
x_test = np.array(x_raw_num)
print(x_test)
# print("Test data shape[1]: {:d}".format(x_test.shape[1]))
# print("输出x_test---------:", list(x_test))
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch         batch_size=64
        batches = data_helpers.batch_iter_pre(list(x_test[:,:88]), FLAGS.batch_size, 1)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            print(x_test_batch.shape)  # (64, 69)
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    # print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)


# 直接调用 sklearn 包
Accu = metrics.accuracy_score(y_test, all_predictions)
print("Accuracy: {:g}".format(Accu))

Recall = metrics.recall_score(y_test, all_predictions, average='macro')
Recall_1 = metrics.recall_score(y_test, all_predictions, average='micro')
print("Recall:{:g}".format(Recall))
print("Recall_1:{:g}".format(Recall_1))

Precision = metrics.precision_score(y_test, all_predictions, average='macro')
print("Precision:{:g}".format(Precision))

F1 = metrics.f1_score(y_test, all_predictions, average='weighted')
print("F1:{:g}".format(F1))
