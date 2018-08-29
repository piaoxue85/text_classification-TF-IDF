# coding:utf-8
import re
import codecs
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import jieba
import pickle
import os
import cut_model

#######################################################################
#数据路径格式
#data_dir
#   |-  label1.txt
#   |-  label2.txt
#   |-  label3.txt


def get_stopwords():
    chinese_num_stop = [u'一', u'二', u'四', u'五', u'六', u'七', u'八', u'九', u'十']
    num_stop = [str(i) for i in range(10)]
    num_stop.extend(chinese_num_stop)
    stopwords_list = [u'【',u'】',u'“',u'”',u'日', u'月', u'的', u'是', u'（', u'）', u'、', u'：', u'，', u'。', u'；', u'《', u'》', '%', '<', '>',
                      '.', ',', ';', '(', ')']
    stopwords_list.extend(num_stop)
    return stopwords_list


def write_model(twm,voca,twm_path,vocabulary_path):
    with open(twm_path, "wb") as twm_obj:
        pickle.dump(twm,twm_obj)
    with open(vocabulary_path, "wb") as voca_obj:
        pickle.dump(voca,voca_obj)


def get_datapath(td_path):
    path = []
    label_list = os.listdir(td_path)
    for l in label_list:
        path.append(td_path+'\\'+l)
    return label_list,path


#构建向量空间
def vector_space(data_path, tdm_path, voca_path):
    label_list, path_list = get_datapath(data_path)
    data = []
    for p in path_list:
        data_file = codecs.open(p, 'r', 'utf-8')
        data.append(data_file.read())

    v = TfidfVectorizer(stop_words=get_stopwords(),sublinear_tf=True)
    tdm = v.fit_transform(data)
    train_vocabulary = v.vocabulary_
    write_model(tdm, train_vocabulary, tdm_path, voca_path)


if __name__ == '__main__':
    # 语料库路径
    data_dir = r'C:\Users\Administrator\Desktop\base_model\stock_words\summary'
    # 分词后语料库路径
    cut_dir = r'E:\PychormProjects\generate\stock_words\train_data'
    if not os.path.exists(cut_dir):
        os.mkdir(cut_dir)
    # 调用分词模块 import cut_model.py
    # 开始分词
    # cut_model.words_cut(data_dir, cut_dir)


    # 模型保存路径
    #tdm_dir:训练数据。
    tdm_dir = r'E:\PychormProjects\generate\stock_words\summary_tdm.dat'
    # vocabulary_dir：向量空间模型
    vocabulary_dir = r'E:\PychormProjects\generate\stock_words\summary_vocabulary.dat'

    print '------generate vector space------'
    vector_space(cut_dir, tdm_dir, vocabulary_dir)
    print '------save model succesfully------'




