import requests
import base64
from PIL import Image
from io import BytesIO


def base64_to_image(base64_string):
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object to work with the image data
    image_buffer = BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_buffer)

    return image


config = {
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "string",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "prompt": "a white house",
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "Euler",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "negative_prompt": "string",
    "eta": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "script_args": [],
    "sampler_index": "Euler",
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {}
}


def predict(prompt: str, model_name: str, num_images: int = 2):
    endpoint = f"http://127.0.0.1:7860/sdapi/v1/sd-models"
    opt = requests.get(endpoint)
    selected_model = None
    if opt.ok:
        models = opt.json()
        for model in models:
            title = model['title']
            if model_name == title.split('/')[0]:
                selected_model = model
        if selected_model:
            endpoint = f"http://127.0.0.1:7860/sdapi/v1/options"
            options_req = requests.get(endpoint)
            if options_req.ok:
                options = options_req.json()
                options['sd_model_checkpoint'] = selected_model['title']
                a = requests.post(url=endpoint, json=options)
                print(a)
            else:
                return {"error": True, "message": "Problem selecting model"}
        else:
            return {"error": True, "message": "Model does not exist or is still in training"}
        endpoint = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        config["prompt"] = prompt
        image_res = requests.post(endpoint, json=config)
        if image_res.ok:
            res_json = image_res.json()
            image = base64_to_image(res_json["images"][0])
            return {"error": False, "image": image}

        else:
            return {"error": True, "message": "Problem generating image"}
    else:
        return {"error": True, "message": "Automatic 1111 may be down"}
