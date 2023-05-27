from fastapi import FastAPI, UploadFile, File
import os
import zipfile
import magic
from train import train
app = FastAPI()


def is_zip_file(file_contents):
    mime_type = magic.from_buffer(file_contents, mime=True)
    print(mime_type)
    return mime_type == 'application/zip'


class NotZipException(Exception):
    pass


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/train")
async def train_dreamooth(model_name: str, class_prompt: str, instance_prompt: str, file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        if not os.path.exists("test123"):
            os.makedirs("datasets")
        if not is_zip_file(file_content):
            raise NotZipException("File uploaded not zip")
        with open(f"datasets/{file.filename}", "wb") as f:
            f.write(file_content)
        unzip_file(f"datasets/{file.filename}", f"datasets/{model_name}")
        is_training = train(model_name=model_name, is_new_model=True,
                            instance_prompt=instance_prompt, class_prompt=class_prompt)
        if is_training:
            return {"error": False, "message": "Model is training"}

        return {"filename": file.filename}
    except FileExistsError:
        return {"message": "File exists"}
    except NotZipException:
        return {"message": "File uploaded not zip"}
