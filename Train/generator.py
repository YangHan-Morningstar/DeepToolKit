import os
import random
import json
import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SequenceGenerator(Sequence):
    def __init__(self, batch_size, target_path, label_num, task_category, maxlen=-1):
        self.batch_size = batch_size
        self.target_path = target_path
        self.label_num = label_num
        self.task_category = task_category
        self.maxlen = maxlen

        self.data_list = os.listdir(target_path)

    def __len__(self):
        return int(np.ceil(len(self.data_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        inputs, outputs = [], []
        row = random.sample(range(0, len(self.data_list)), self.batch_size)
        for i in row:
            with open(self.target_path + "/" + self.data_list[i]) as temp_file:
                temp_dict = json.load(temp_file)
            inputs.append(temp_dict["data_features"])
            outputs.append(temp_dict["label"])
        if self.task_category == "nlp":
            inputs = np.array(pad_sequences(inputs, maxlen=self.maxlen, padding="post"))
        elif self.task_category == "cv":
            inputs = np.array(inputs)
        outputs = np.array(to_categorical(outputs, self.label_num))
        return inputs, outputs


class GeneralGenerator(object):

    def train_data_generator(self, batch_size, target_path, label_num, task_category, maxlen=-1):
        while True:
            inputing, outputing = [], []
            data_list = os.listdir(target_path)
            row = random.sample(range(0, len(data_list)), batch_size)
            for i in row:
                with open(target_path + "/" + data_list[i]) as temp_file:
                    temp_dict = json.load(temp_file)

                inputing.append(temp_dict["data_features"])
                outputing.append(temp_dict["label"])

            if task_category == "nlp":
                inputing = np.array(pad_sequences(inputing, maxlen=maxlen, padding="post"))
            elif task_category == "cv":
                inputing = np.array(inputing)

            outputing = np.array(to_categorical(outputing, label_num))

            yield inputing, outputing

    def val_data_generator(self, batch_size, target_path, label_num, task_category, maxlen=-1):
        while True:
            inputing, outputing = [], []
            data_list = os.listdir(target_path)
            row = random.sample(range(0, len(data_list)), batch_size)
            for i in row:
                with open(target_path + "/" + data_list[i]) as temp_file:
                    temp_dict = json.load(temp_file)

                inputing.append(temp_dict['data_features'])
                outputing.append(temp_dict['label'])

            if task_category == "nlp":
                inputing = np.array(pad_sequences(inputing, maxlen=maxlen, padding="post"))
            elif task_category == "cv":
                inputing = np.array(inputing)

            outputing = np.array(to_categorical(outputing, label_num))

            yield inputing, outputing
