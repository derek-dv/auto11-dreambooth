import requests
import json

url = "http://127.0.0.1:7860"


def train(model_name: str = "Test", is_new_model: bool = True, class_prompt="", instance_prompt="", scheduler: str = "DEISMultistep", src_model: str = "/workspace/stable-diffusion-webui/models/Stable-diffusion/SDv1-5.ckpt"):
    # Create model
    if is_new_model:
        create_endpoint = f"{url}/dreambooth/createModel?new_model_name={model_name}&new_model_src={src_model}&is_512=true&new_model_extract_ema=false&new_model_scheduler={scheduler}"
        print(create_endpoint)
        create_res = requests.post(create_endpoint)
        text = create_res.text
        print(text)

    with open("sample_config.json", "r") as f:
        txt = f.read()
        config = json.loads(txt)
    config["model_name"] = model_name
    config["model_dir"] = f"/workspace/stable-diffusion-webui/models/dreambooth/{model_name}"
    config["model_path"] = f"/workspace/stable-diffusion-webui/models/dreambooth/{model_name}"
    config[
        "pretrained_model_name_or_path"] = f"/workspace/stable-diffusion-webui/models/dreambooth/{model_name}/working"
    config["concepts_list"][0]["class_prompt"] = class_prompt
    config["concepts_list"][0][
        "instance_data_dir"] = f"/workspace/stable-diffusion-webui/datasets/{model_name}"
    config["concepts_list"][0]["instance_prompt"] = instance_prompt
    config["scheduler"] = "EulerAncestralDiscrete"

    config_endpoint = f"{url}/dreambooth/model_config"
    config_res = requests.post(
        config_endpoint,
        json=config
    )
    print("Config added")
    print(config_res.text)
    train_endpoint = f'{url}/dreambooth/start_training?model_name={model_name}&use_tx2img=true'
    train_res = requests.post(train_endpoint)
    print(train_res.text)
    if train_res.status_code == 200:
        return True
    return False

    # Add config
