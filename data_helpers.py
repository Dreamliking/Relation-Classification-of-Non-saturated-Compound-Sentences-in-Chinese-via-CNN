# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xlrd
import tensorflow as tf
import re
import itertools
import codecs
from collections import Counter
import jieba
import pynlpir
import pickle
import json
import os
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

# 路径保存,使用停用词表
_PAD_ID = 0
# BL_PATH = './data/two_cccs_lack_generalized_bl_train02.txt'
# YG_PATH = './data/two_cccs_lack_generalized_yg_train02.txt'
# ZZ_PATH = './data/two_cccs_lack_generalized_zz.txt'
STOP_PATH = './data/stopword1208.txt'
Model_path = './train_file/word_embedding2_stop.model'
relative_path = './relative_data/relative.txt'

# 数据集分开
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15]):
    # number of examples
    data_len = len(x)
    lens = [int(data_len*item) for item in ratio]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)

def seg_word(file_path):
    Big_list = []
    with open(file_path, "r", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = str(line)
            line = list(jieba.cut(line))
            # print(line)
            Big_list.append(line)
    return Big_list


def load_data_and_labels(bl_data_file, yg_data_file, zz_data_file):
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  """

  bl_examples = seg_word(bl_data_file)
  yg_examples = seg_word(yg_data_file)
  zz_examples = seg_word(zz_data_file)

  # 两个标签
  x_text = bl_examples + yg_examples + zz_examples

  # x_text = [clean_str(sent) for sent in x_text]
  # x_text = [list(s) for s in x_text]

  bl_labels = [[1, 0, 0] for _ in bl_examples]
  yg_labels = [[0, 1, 0] for _ in yg_examples]
  zz_labels = [[0, 0, 1] for _ in zz_examples]
  y = np.concatenate([bl_labels, yg_labels, zz_labels], 0)

  return [x_text, y]


# 将句子统一长度
def pad_sentences(sentences, padding_word="<PAD/>"):
  """
  Pads all sentences to the same length. The length is defined by the longest sentence.
  Returns padded sentences.
  """
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences


# 建立词典需要的函数
def _read_words(path, stop_word_flag=False, read_to_sentences=False):
    """ 给定文本的path， 把所有分割好的token读出
    sentences_flag : True  -> 返回[[sentence 0 ]...[sentence 99 ] ... ],
                     False -> 返回[word0, word2,......word999999...] (将句子截断)
    """
    data = []
    with tf.gfile.GFile(path, mode='r') as file:
        for line in file:
            sent = line.strip()
            if not stop_word_flag:
                words = jieba.cut(sent)
                if read_to_sentences:
                    data.append(words)
                else:
                    for word in words:
                        data.append(word)
            else:
                data.append(sent)
    return data


# 将生成的word embedding存到文件中 这个和停用词没有什么关系，得到的预训练词向量是不变的
def file_to_wordembedding(bl_path, yg_path, zz_path):
    vector_path = './data/cccs_vector2.txt'
    total_word_path = './data/total_word2.txt'

    bl_word = seg_word(bl_path)
    yg_word = seg_word(yg_path)
    zz_word = seg_word(zz_path)

    total_word = bl_word + yg_word + zz_word

    with open(total_word_path, 'w') as file:
        for sentence in total_word:
            file.write(" ".join(sentence))    # 预训练词向量,生成与对应词典的词向量
    model = Word2Vec(LineSentence(total_word_path), size=128, window=5, min_count=5, workers=multiprocessing.cpu_count() - 4)
    # model.wv.save_word2vec_format(vector_path)
    model.save(Model_path)


# 将生成的词典存到文件中
def file_to_vocab(bl_path, yg_path, zz_path, stop_path):
    vocab_path = "./stop_data/vocabulary_stop2.pickle"
    reversed_vocab_path = "./stop_data/reversed_vocabulary_stop2.pickle"
    # 词表存在则直接加载
    if os.path.exists(vocab_path) and os.path.exists(reversed_vocab_path):
        with open(vocab_path, "rb") as file:
            with open(reversed_vocab_path, "rb") as re_file:
                return pickle.load(file), pickle.load(re_file)

    print("下面开始制作词表")
    bl_words = _read_words(bl_path)
    yg_words = _read_words(yg_path)
    zz_words = _read_words(zz_path)
    stop_words = _read_words(stop_path, stop_word_flag=True)
    relative_words = _read_words(relative_path)

    # 开始去停用词
    total_words = bl_words + yg_words + zz_words
    print("total length", len(total_words))
    total_words = [word for word in total_words if word not in (stop_words and relative_words)]

    # collections.Counter 搞不定Unicode码， 自己手动统计词频
    count_pairs = {}
    # illegal_words = []  # 不在预训练模型内的词，先不考虑预训练模型
    for word in total_words:
        if word in count_pairs.keys():
            count_pairs[word] += 1
        else:
            count_pairs[word] = 1
            print(word, "进入词表")

    # with open("./data/illegal_words.txt", mode="w") as file:
    #     for line in illegal_words:
    #         file.write("".join(line) + "\n")

    # 把统计词频后的dict排序， 砍掉很低频的词和高频的词，留下5000个词
    sorted_pairs = sorted(count_pairs.items(), key=lambda item: item[1])
    # vocab_pairs = sorted_pairs[-5050: -50]
    # vocab_pairs = sorted_pairs[-3050: -50]
    vocab_pairs = sorted_pairs
    # 这边这么打算： 不采用UNK， 扔掉所有不在词表里面的词，
    # 给每个token按序号分配id PAD（填充词）放进表的头部
    vocab_dic = dict((vocab_pairs[i][0], i+1) for i in range(len(vocab_pairs)))
    vocab_dic["<PAD/>"] = _PAD_ID  # 0

    # print("输出字典：", vocab_dic)
    # 将词表字典序列化保存起来。
    with tf.gfile.GFile(vocab_path, mode="w") as file:
        pickle.dump(vocab_dic, file)
    # 把字典放入json
    with open("./stop_data/vocab_stop2.json", 'w') as file:
        dict_json = json.dump(vocab_dic, file)

    # with open("./data/vocab_stop.txt", 'w') as file:
    #     dict_txt = file.write(vocab_dic.keys())

    # 将词典翻转之后保存起来: {词 : id } -> {id : 词}
    reversed_vocab = {id: token for token, id in vocab_dic.items()}
    with tf.gfile.GFile(reversed_vocab_path, mode="w") as file:
        pickle.dump(reversed_vocab, file)
    return [vocab_dic, reversed_vocab]


# 创建一个关系词和矩阵的词典
def Word_mark_to_dic(relative_mark_path):
    Word_mark_vocab_path = './relative_data/word_mark_stop2.pickle'

    # 词表存在则直接加载
    if os.path.exists(Word_mark_vocab_path):
        with open(Word_mark_vocab_path, "rb") as file:
            return pickle.load(file)

    list1 = ['bl', 'dj', 'js', 'jz', 'lg', 'md', 'rz', 'td', 'tj', 'xz', 'yg', 'zz']
    Matrix = np.eye(12)
    list2 = Matrix.tolist()
    classification_dict = dict(zip(list1, list2))

    data = xlrd.open_workbook(relative_mark_path)
    table = data.sheet_by_name('relative_mark')
    list6 = table.col_values(2)[1:]

    list7 = []
    list_1 = []
    for index, item in enumerate(table.col_values(0)[1:]):
        if re.match(r'\d{1}', item[-1:]):
            item = item[:-1]
        list7.append(item)
    for element in list6:
        list_1.append(classification_dict.get(element))

    Word_mark = {}
    for i in range(len(list7)):
        if i == 0:
            Word_mark[list7[i]] = list_1[i]
        else:
            if list7[i] == list7[i - 1]:
                Word_mark[list7[i - 1]] = np.array(Word_mark[list7[i - 1]]) + np.array(list_1[i])
            else:
                Word_mark[list7[i]] = list_1[i]
    with tf.gfile.GFile(Word_mark_vocab_path, mode="w") as file:
        pickle.dump(Word_mark, file)
    return Word_mark


# 判断词典中的词是否在关系词表中，如果在则标记为1,接着找到关系词所对应的关系标记的矩阵
# 最终返回的是一个矩阵 13维的
def add_feature(vocab_dic, relative_path, relative_mark_path):
    # 定义13维的矩阵
    Relative = np.zeros([len(vocab_dic), 13])
    relative_words = []
    with open(relative_path, encoding='utf-8') as file:
        for line in file.readlines():
            relative_words.append(line.strip())
        # print(relative_words)
    for word, index in vocab_dic.items():
        if word in relative_words:
            Relative[index:index+1, 0] = 1
            Word_mark = Word_mark_to_dic(relative_mark_path)
            Relative[index, 1:] = Word_mark.get(word, 0)
            print("{}:{} = {}".format(index, word, Relative[index]))
        else:
            Relative[index:index+1, 0] = 0
        # print("{}:{} = {}".format(index, word, Relative[index]))
    return Relative


# 词典映射到词向量   词序列
def word_to_embedding(Model_path, wordfile, vocab_dic,  relative_path, relative_mark_path):
    word_embedding = np.zeros([len(vocab_dic), 128])
    # model = gensim.models.KeyedVectors.load_word2vec_format(Model_path)
    model = gensim.models.KeyedVectors.load(Model_path)
    for word, index in vocab_dic.items():
        if word in model.wv.vocab:
            word_embedding[index] = model[word]
        if word == ["<PAD/>"]:
            word_embedding[index] = [random.uniform(-1, 1) for index in range(128)]
        # print("{}:{} = {}".format(index, word, word_embedding[index]))
    Relative = add_feature(vocab_dic,  relative_path, relative_mark_path)
    word_embedding = np.hstack((word_embedding, Relative))
    np.save(wordfile, word_embedding)


def build_input_data(sentences, labels, vocab_dic):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocab_dic.get(word, 0) for word in sentence] for sentence in sentences])
  y = np.array(labels)
  return [x, y]

# def build_embedding():


def load_data(bl_data_file, yg_data_file, zz_data_file, stop_word_file):
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, labels = load_data_and_labels(bl_data_file, yg_data_file, zz_data_file)
  sentences_padded = pad_sentences(sentences)
  vocab_dic, reversed_vocab = file_to_vocab(bl_data_file, yg_data_file, zz_data_file, stop_word_file)
  x, y = build_input_data(sentences_padded, labels, vocab_dic)
  return [x, y, vocab_dic, reversed_vocab]


# def load_data(bl_data_file, yg_data_file, zz_data_file):
#   """
#   Loads and preprocessed data for the MR dataset.
#   Returns input vectors, labels, vocabulary, and inverse vocabulary.
#   """
#   # Load and preprocess data
#   sentences, labels = load_data_and_labels(bl_data_file, yg_data_file, zz_data_file)
#   sentences_padded = pad_sentences(sentences)
#   vocabulary, vocabulary_inv = build_vocab(sentences_padded)
#   x, y = build_input_data(sentences_padded, labels, vocabulary)
#   return [x, y, vocabulary, vocabulary_inv]

def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  生成数据集
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]


def batch_iter_pre(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    生成数据集
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    # Shuffle the data at each epoch

    # shuffle_indices = np.random.permutation(np.arange(data_size))
    # shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
