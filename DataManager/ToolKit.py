import json
import random
import os
import glob
import shutil


class ToolKit(object):

    def __init__(self, source_data_filepath):
        self.source_data_filepath = source_data_filepath
        self.label_dict = {}
        self.filepath_label_dict = {}
        self.data_counter = 0
        self.train_data_counter = 0
        self.val_data_counter = 0

    # 得到数据绝对路径列表与种类列表（两者一一对应）
    def get_data_filepath_and_label(self):

        data_category_list = os.listdir(self.source_data_filepath)
        filepath_list, label_list = [], []

        for category in data_category_list:
            self.add_label_dict(category)

            data_filepath_list = glob.glob(self.source_data_filepath + "/" + category + "/*")
            for data_filepath in data_filepath_list:
                self.data_counter += 1
                filepath_list.append(data_filepath)
                label_list.append(category)

        return filepath_list, label_list

    # 得到数据的绝对路径-标签词典
    def get_filepath_label_dict(self):
        print("Getting filepath and label dict...")

        filepath_list, label_list = self.get_data_filepath_and_label()

        self.shuffle_data(filepath_list, label_list)

        if self.filepath_label_dict != {}:
            self.filepath_label_dict = {}
        for i in range(len(filepath_list)):
            self.filepath_label_dict[filepath_list[i]] = self.label_dict[label_list[i]]

    # 扩充标签词典
    def add_label_dict(self, category):
        if category not in self.label_dict:
            self.label_dict[category] = len(self.label_dict)

    # 获取标签词典
    def get_label_dict(self):
        return self.label_dict

    # 将数据标签词典写入json文件
    def write_to_json(self, path, data_label_dict):
        with open(path, 'w') as json_file:
            json_file.write(json.dumps(data_label_dict))

    def read_from_json(self, path):
        with open(path) as temp_file:
            temp_dict = json.load(temp_file)
        return temp_dict

    def write_to_txt(self, path, content, encoding="utf-8"):
        with open(path, 'w', encoding=encoding) as file:
            file.write(content)

    # 随机打乱数据
    def shuffle_data(self, data, label):
        temp_data, temp_label = [], []
        random_num_array = random.sample(range(0, len(data)), len(data))

        for i in random_num_array:
            temp_data.append(data[i])
            temp_label.append(label[i])

        data, label = temp_data, temp_label

    # 将完整的数据集中每一类按比例划分为训练集和验证集
    def segmentation(self, source_data_filepath, target_filepath, rate=0.2):
        print("Automatically segmenting data...")
        target_filepath_train = target_filepath + "/train"
        target_filepath_val = target_filepath + "/val"

        if not os.path.exists(target_filepath_train):
            os.makedirs(target_filepath_train)
        if not os.path.exists(target_filepath_val):
            os.makedirs(target_filepath_val)

        for category in data_category_list:
            data_filepath_list = glob.glob(source_data_filepath + "/" + category + "/*")
            this_data_train_num = int(len(data_filepath_list) * (1 - rate))
            if this_data_train_num == 0:
                this_data_train_num += 1
                print("Warning! The " + category + " data is too little!")
            train_data_filepath_list = data_filepath_list[0: this_data_train_num - 1]
            val_data_filepath_list = data_filepath_list[this_data_train_num:]

            if not os.path.exists(target_filepath_train + "/" + category):
                os.makedirs(target_filepath_train + "/" + category)
            if not os.path.exists(target_filepath_val + "/" + category):
                os.makedirs(target_filepath_val + "/" + category)

            for each_filepath in train_data_filepath_list:
                shutil.copy(each_filepath, target_filepath_train + "/" + category)

            for each_filepath in val_data_filepath_list:
                shutil.copy(each_filepath, target_filepath_val + "/" + category)

        print("Finish!")

        return self.train_data_counter, self.val_data_counter

    def get_data_counter(self):
        return self.data_counter

    def get_file_num_in_path(self, filepath):
        sub_file_list = os.listdir(filepath)
        return len(sub_file_list)

    def check_target_path(self, target_path):
        if not os.path.exists(target_path):
            os.makedirs(target_path)
