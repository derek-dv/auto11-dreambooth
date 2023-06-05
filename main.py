from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse
import os
from io import BytesIO
import zipfile
from train import train
from predict import predict
app = FastAPI()


def is_zip_file(file_path):
    try:
        with zipfile.ZipFile(file_path) as zip_file:
            return True
    except zipfile.BadZipFile:
        return False


class NotZipException(Exception):
    pass


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/train")
async def train_dreamooth(file: UploadFile, model_name: str = Body(...), instance_prompt: str = Body(...), class_prompt: str = Body(...), is_new_model: bool = Body(False)):
    try:
        print(file)
        if is_new_model:
            file_content = await file.read()
            if not os.path.exists(f"/workspace/stable-diffusion-webui/datasets/{model_name}"):
                os.makedirs(
                    f"/workspace/stable-diffusion-webui/datasets/{model_name}")
            with open(f"/workspace/stable-diffusion-webui/datasets/{file.filename}", "wb") as f:
                f.write(file_content)
            if not is_zip_file(f"/workspace/stable-diffusion-webui/datasets/{file.filename}"):
                raise NotZipException("File uploaded not zip")
            unzip_file(f"/workspace/stable-diffusion-webui/datasets/{file.filename}",
                       f"/workspace/stable-diffusion-webui/datasets/{model_name}")
        is_training = train(model_name=model_name, is_new_model=is_new_model,
                            instance_prompt=instance_prompt, class_prompt=class_prompt)
        if is_training:
            return {"error": False, "message": "Training started"}

        return {"filename": file.filename}
    except FileExistsError:
        print("Exists")
        return {"message": "File exists"}
    except NotZipException:
        return {"message": "File uploaded not zip"}


@app.post("/predict")
async def predict_dreambooth(model_name: str = Body(...), prompt: str = Body(...)):
    result = predict(prompt, model_name)
    if not result["error"]:
        image = result["image"]
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        return StreamingResponse(image_bytes, media_type="image/png")

    return result
