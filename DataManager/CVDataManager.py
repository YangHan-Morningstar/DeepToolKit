from DataManager.ToolKit import ToolKit
import cv2 as cv


class CVDataManager(ToolKit):

    def __init__(self, source_data_filepath, target_path):
        super().__init__(source_data_filepath)
        self.filepath_label_dict = {}
        self.target_path = target_path

    # 处理数据，提取基础特征
    def extract_feature_just_rgb(self, img_rows=224, img_cols=224):
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

    # 得到数据的绝对路径-标签词典
    def get_filepath_label_dict(self):
        print("Getting filepath and label dict...")

        filepath_list, label_list = self.get_data_filepath_and_label()

        self.shuffle_data(filepath_list, label_list)

        for i in range(len(filepath_list)):
            self.filepath_label_dict[filepath_list[i]] = self.label_dict[label_list[i]]

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


if __name__ == '__main__':
    # print("Please input the source data file path")
    source_data_filepath = "/root/Artist/images/images"
    target_path = "/root/Artist/json_data/RGB_Feature"
    CVDataManager = CVDataManager(source_data_filepath=source_data_filepath, target_path=target_path)
    CVDataManager.get_filepath_label_dict()
    CVDataManager.extract_feature_just_rgb(img_rows=299, img_cols=299)
    print("Finish!")
