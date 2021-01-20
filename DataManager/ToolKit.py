import json
import random
import os
import glob


class ToolKit(object):

    def __init__(self, source_data_file_path):
        self.source_data_file_path = source_data_file_path
        self.label_dict = {}
        self.data_counter = 0

    # 得到数据绝对路径列表与种类列表（两者一一对应）
    def get_data_filepath_and_label(self):

        data_category_list = os.listdir(self.source_data_file_path)
        filepath_list, label_list = [], []

        for category in data_category_list:
            self.add_label_dict(category)

            data_filepath_list = glob.glob(self.source_data_file_path + "/" + category + "/*")
            for data_filepath in data_filepath_list:
                self.data_counter += 1
                filepath_list.append(data_filepath)
                label_list.append(category)

        return filepath_list, label_list

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

    # 随机打乱数据
    def shuffle_data(self, data, label):
        temp_data, temp_label = [], []
        random_num_array = random.sample(range(0, len(data)), len(data))

        for i in random_num_array:
            temp_data.append(data[i])
            temp_label.append(label[i])

        return data, label
