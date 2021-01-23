from DataManager.ToolKit import ToolKit
import cv2 as cv
import os


class CVDataManager(ToolKit):

    def __init__(self, source_data_filepath="", target_path=""):
        super().__init__(source_data_filepath)
        self.target_path = target_path

        if not os.path.exists(self.target_path) and self.target_path != "":
            os.makedirs(self.target_path)

    # 处理数据，提取基础特征
    def extract_feature_just_rgb(self, img_rows=224, img_cols=224, reset=1):
        print("Extracting RGB feature from image...")
        counter = 0
        error_file_counter = 0
        for data_filepath, label in self.filepath_label_dict.items():
            features_label_dict = {}
            normalized_feature, error = self.normalize(data_filepath, img_rows, img_cols)

            if error == 0:
                features_label_dict["data_features"] = normalized_feature.tolist()
                features_label_dict["label"] = label

                self.write_to_json(self.target_path + "/" + str(counter) + ".json", features_label_dict)

                counter += 1

            error_file_counter += error

        print("There are " + str(error_file_counter) + " file loading fail.")

        if reset == 1:
            self.filepath_label_dict = {}

    # 读取图片并标准化
    def normalize(self, data_filepath, img_rows, img_cols):
        error = 0
        normalized_feature = []
        try:
            original_feature = cv.imread(data_filepath)
            resize = cv.resize(original_feature, (img_rows, img_cols))
            normalized_feature = cv.normalize(resize, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        except:
            error = 1
        return normalized_feature, error

    def set_source_data_filepath(self, source_data_filepath):
        self.source_data_filepath = source_data_filepath

    def set_target_filepath(self, target_filepath):
        self.target_path = target_filepath

        if not os.path.exists(self.target_path) and self.target_path != "":
            os.makedirs(self.target_path)


if __name__ == '__main__':
    # print("Please input the source data file path")
    source_data_filepath = "/root/Artist/images/images"
    target_path = "/root/Artist/json_data/RGB_Feature"
    CVDataManager = CVDataManager(source_data_filepath=source_data_filepath, target_path=target_path)
    CVDataManager.get_filepath_label_dict()
    CVDataManager.extract_feature_just_rgb(img_rows=299, img_cols=299)
    print("Finish!")
