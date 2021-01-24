import os
import random
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Generator(object):

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
