import yaml
import os

def open_config(config_file_name="config.yaml"):
    with open(config_file_name, 'r') as config:
        print(yaml.safe_load(config))


def get_root_img_folder(config_file_name="config.yaml"):
    config = open_config(config_file_name=config_file_name)
    return config["target_feature"] + "_" + config["sample_number"] + "_data"


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


def get_device(config_file_name):
    config = open_config(config_file_name=config_file_name)
    return config["device"]


def does_need_train(config_file_name):
    config = open_config(config_file_name=config_file_name)
    return config["train"]
