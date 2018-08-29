# coding:utf-8
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import cut_model
import jieba
import string

dict_label = {'cai_wu': u'财务', 'fa_lv': u'法律', 'FinancialGlossary': u'英文财经', 'guan_li': u'管理', 'ji_jin': u'基金'
              , 'jin_rong': u'金融', 'jing_ji': u'经济', 'jing_ji_li_lun': u'经济理论', 'kuai_ji': u'会计'
              , 'qu_wei_ji_rong': u'趣味金融', 'tou_zi': u'投资', 'zhai_quan':u'债券'
              , 'zheng_quan': u'证券', 'zi_xun': u'咨询', 'zong_he': u'综合'}


def get_stopwords():
    chinese_num_stop = [u'一', u'二', u'四', u'五', u'六', u'七', u'八', u'九', u'十']
    num_stop = [str(i) for i in range(10)]
    num_stop.extend(chinese_num_stop)
    stopwords_list = [u'【', u'】', u'“', u'”', u'日', u'月', u'的', u'是', u'（', u'）',u'?',u'、', u'：', u'，', u'。', u'；', u'《',
                      u'》', '%', '<', '>', '.', ',', ';', '(', ')']
    stopwords_list.extend(num_stop)
    return stopwords_list


def read_modle(twm_path, vocabulary_path):
    with open(twm_path, 'rb') as twm_obj:
        train_twm = pickle.load(twm_obj)
    with open(vocabulary_path, 'rb') as voca_obj:
        train_vocabulary = pickle.load(voca_obj)
    return train_twm, train_vocabulary


def get_datapath(td_path):
    path = []
    label_list = []
    label_name = os.listdir(td_path)
    for la in label_name:
        path.append(td_path+'\\'+la)
        label_list.append(la.split('.txt')[0])
    return label_list, path


def predict_label(predict_path, train_path, tdm_path, voca_path):
    labels_list, _ = get_datapath(train_path)
    test_label_list, test_label_path = get_datapath(predict_path)
    # stopwords = get_stopwords()
    print ('------load TF-IDF model------')
    train_data, train_vocabulary = read_modle(tdm_path, voca_path)

    print('------begin construct model------')
    # 加载模型
    v = TfidfVectorizer(sublinear_tf=True, vocabulary=train_vocabulary)

    print '------begin predict------'
    clf = svm.SVC()
    clf.fit(train_data, labels_list)

    pred_list = []
    for i in range(len(test_label_path)):
        data = []
        print("pridict: ", test_label_list[i])
        data_file = codecs.open(test_label_path[i], 'r', 'utf-8')
        data.append(data_file.read())
        predict_data = v.fit_transform(data)
        pred = clf.predict(predict_data)
        pred_list.append(pred[0])
        # pred_one_list = []
        '''for ol in data_file:
            data = []
            data.append(ol)
            predict_data = v.fit_transform(data)
            pred = clf.predict(predict_data)
            pred_one_list.append(pred[0])'''
        # pred_list.append(pred_one_list)
    print '-------predict succesfully--------'
    return labels_list, pred_list


if __name__ == '__main__':
    # 语料库路径
    predict_data_dir = 'test_data'
    # 分词后预料库路径
    train_cut_dir = r'train_data'

    # 加载向量空间模型
    tdm_dir = 'summary_tdm.dat'
    vocabulary_dit = 'summary_vocabulary.dat'
    # 输出标签
    label_out_path = r'C:\Users\Administrator\Desktop\base_model\predict_labels.txt'
    # 停用词路径
    # stopwords_dir = r'C:\Users\Administrator\Desktop\my_demo_data\stop_words.txt'

    label_list, pre_label = predict_label(predict_data_dir, train_cut_dir, tdm_dir, vocabulary_dit)

    print label_list, '\n', pre_label
    cun = 0.0
    all_cun = 0.0
    for i in range(len(label_list)):
        all_cun += 1
        if pre_label[i] == label_list[i]:
                cun += 1
    print cun, all_cun, cun / all_cun
