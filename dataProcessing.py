import collections
import re
import random
import numpy as np
import hyperparameters



class Inst(object):  # 定义Inst类型，将文件中的每句话生成Inst的数据类型
    def __init__(self):
        self.m_word = []  # 将一句话中除label外的每个单词都存入m_word列表
        self.m_label = 0  # 将一句话中的label存入m_label中
        # def show(self):
        #     print(self.m_word)
        #     print(self.m_label)


class vocab(object):  # 生成词典
    def __init__(self):
        self.v_list = []
        self.v_dict = collections.OrderedDict()  # 固定字典

    def MakeVocab(self, ListName):  # 生成词典的过程
        for i in range(len(ListName)):
            if ListName[i] not in self.v_list:
                self.v_list.append(ListName[i])
        self.v_list.append("-unknown-")
        for n in range(len(self.v_list)):
            self.v_dict[self.v_list[n]] = n

        return self.v_dict


class example(object):  #
    def __init__(self):
        self.word_indexes = []
        self.label_index = []

class Reader():
    def readfile(self, path):
        f = open(path, 'r')
        newList = []
        count = 0
        for line in f.readlines():
            count += 1
            new = Inst()
            # x = line.strip().split("||| ")  # big bug big bug big bug stupid bug stupid bug
            label, seq, sentence = line.partition(" ")
            sentence = Reader.clean_str(sentence)
            new.m_word.append(sentence.split(" "))
            new.m_label = label
            newList.append(new)

            if count == -1:
                break
        random.seed(0)
        random.shuffle(newList)
        f.close()
        return newList[:int(len(newList) * 0.7)], \
               newList[int(len(newList) * 0.7):int(len(newList) * 0.8)],\
               newList[int(len(newList) * 0.8):]

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip()


class Classifier(object):

    def __init__(self):
        self.testLabel = vocab()
        self.testWord = vocab()
        self.hyperparameter = hyperparameters.HyperParameters()

    def ToList(self, InitList):
        wordList = []
        labelList = []
        for i in range(len(InitList)):
            for j in InitList[i].m_word[0]:
                wordList.append(j)
            labelList.append(InitList[i].m_label)
        print("训练集WordList:",len(wordList))
        return wordList, labelList

    def SentenceInNum(self, initRst, word_dict, label_dict):
        example_list = []
        for i in range(len(initRst)):
            dist = example()
            for j in initRst[i].m_word[0]:
                if j in word_dict:
                    id = word_dict[j]
                else:
                    id = word_dict["-unknown-"]
                dist.word_indexes.append(id)
            num = label_dict[initRst[i].m_label]
            dist.label_index.append(num)
            example_list.append(dist)
        return example_list

    def load_my_vecs(self, path, vocab, freqs, k=None):
        word_vecs = {}
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                # word = word.lower()
                # if word in vocab and freqs[word] != 1:  # whether to judge if in vocab
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    # if word in vocab:  # whether to judge if in vocab
                    #     if count % 5 == 0 and freqs[word] == 1:
                    #         continue
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs


    def add_unknown_words_by_uniform(self, word_vecs, vocab, k=100):
        list_word2vec = []
        oov = 0
        iov = 0
        # uniform = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                # word_vecs[word] = np.random.uniform(-0.1, 0.1, k).round(6).tolist()
                # word_vecs[word] = uniform
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        print("oov count", oov)
        print("iov count", iov)

        return list_word2vec


    def firstDataProcessing(self, data, hy):
        reader = Reader()
        initRst_train, initRst_dev, initRst_test = reader.readfile(data)
        (wordList_train, labelList_train) = self.ToList(initRst_train)

        word_dict = self.testWord.MakeVocab(wordList_train)
        label_dict = self.testLabel.MakeVocab(labelList_train)

        label_dict.pop("-unknown-")

        word2vec = self.load_my_vecs(path=self.hyperparameter.word_Embedding_Path,
                                     vocab=word_dict, freqs=None, k=300)

        self.hyperparameter.pretrained_weight = self.add_unknown_words_by_uniform(word_vecs=word2vec,
                                                                                  vocab=word_dict, k=300)
        Example_list_dev = self.SentenceInNum(initRst_dev, word_dict, label_dict)
        Example_list_test = self.SentenceInNum(initRst_test, word_dict, label_dict)
        Example_list_train = self.SentenceInNum(initRst_train, word_dict, label_dict)

        hy.unknown = word_dict["-unknown-"]
        hy.embedding_num = self.hyperparameter.unknown + 1
        hy.labelSize = len(label_dict)

        return Example_list_train, Example_list_dev, Example_list_test, initRst_test
