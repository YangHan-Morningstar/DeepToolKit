from DataManager.ToolKit import ToolKit
import os
import jieba


class NLPDataManager(ToolKit):

    def __init__(self, source_data_filepath="", target_path=""):
        super().__init__(source_data_filepath)
        self.target_path = target_path
        self.word_frequency = {}
        self.word_dict = {"NaN": 0}
        self.text_max_len = 0

        if not os.path.exists(self.target_path) and self.target_path != "":
            os.makedirs(self.target_path)

    def word_frequency_statistics(self):
        print("Getting word frequency...")

        for filepath, label in self.filepath_label_dict.items():
            txt_content = self.get_txt_content_from_text_file(filepath)
            word_list = self.cut_sentence_to_word_and_cal_maxlen(txt_content)

            for word in word_list:
                if word not in self.word_frequency:
                    self.word_frequency[word] = 1
                else:
                    self.word_frequency[word] += 1

    def cal_word_dict(self, top_k=5):
        print("Calculating word dict...")

        for word, fre in self.word_frequency.items():
            if fre > top_k:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

    def get_word_dict(self):
        return self.word_dict

    def set_word_dict(self, word_dict_filepath):
        self.word_dict = self.read_from_json(word_dict_filepath)

    def get_txt_max_len(self):
        return self.text_max_len

    def extract_feature_just_by_dict(self):
        print("Extracting feature by dict...")

        counter = 0
        for filepath, label in self.filepath_label_dict.items():
            txt_content_num = []
            features_label_dict = {}
            txt_content = self.get_txt_content_from_text_file(filepath)
            word_list = self.cut_sentence_to_word_and_cal_maxlen(txt_content)

            for word in word_list:
                if word in self.word_dict:
                    txt_content_num.append(self.word_dict[word])
                else:
                    txt_content_num.append(self.word_dict["NaN"])

            features_label_dict["data_features"] = txt_content_num
            features_label_dict["label"] = label

            self.write_to_json(self.target_path + "/" + str(counter) + ".json", features_label_dict)

            counter += 1

    def get_txt_content_from_text_file(self, txt_filepath):
        f = open(txt_filepath, 'r')
        txt_content = f.read()
        f.close()
        return txt_content

    def cut_sentence_to_word_and_cal_maxlen(self, sentence):
        temp_list = []
        word_list = jieba.cut(sentence, cut_all=False, HMM=True)
        for word in word_list:
            temp_list.append(word)
        self.text_max_len = max(self.text_max_len, len(temp_list))
        return temp_list

    def set_source_data_filepath(self, source_data_filepath):
        self.source_data_filepath = source_data_filepath

    def set_target_filepath(self, target_filepath):
        self.target_path = target_filepath

        if not os.path.exists(self.target_path) and self.target_path != "":
            os.makedirs(self.target_path)

    def reset(self, source_data_filepath="", target_path=""):
        self.source_data_filepath = source_data_filepath
        self.target_path = target_path
        self.label_dict = {}
        self.filepath_label_dict = {}
        self.data_counter = 0
        self.train_data_counter = 0
        self.val_data_counter = 0
        self.word_frequency = {}
        self.word_dict = {"NaN": 0}
        self.text_max_len = 0

        if not os.path.exists(self.target_path) and self.target_path != "":
            os.makedirs(self.target_path)
