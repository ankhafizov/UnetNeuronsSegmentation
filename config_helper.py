import yaml
import os

def open_config(config_file_name="config.yaml"):
    with open(config_file_name, 'r') as config:
        return yaml.safe_load(config)


def get_root_img_folder(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["target_feature"] + "_"\
           + str(config["sample_number"]) + "_data"


def get_input_tomo_img_folder(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["input_tomo_images"]


def get_mask_img_folder(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    root = get_root_img_folder(config_file_name=config_file_name)
    return os.path.join(root, config["mask_images"])


def get_OUTPUT_masked_img_folder(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    root = get_root_img_folder(config_file_name=config_file_name)
    return os.path.join(root, config["output_masked_images"])


def get_model_name(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return "MODEL_" + config["target_feature"] + ".pth"


def get_device(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["device"]


def get_start_prediction_point(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    point = config["start_predictions"]
    point = 0 if point == "beginning" else point

    return point


def does_need_train(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["train"]


def does_need_predict(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["predict"]


def does_need_cleaning(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["cleaning"]