import os
from google.cloud import storage 
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.models.predict_model import eval_gcs

# Create a client
client = storage.Client()

# Get the bucket
bucket = client.bucket("butterfly_jar")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# http://127.0.0.1:8000/generate_sample/?steps=1&n_images=1&seed=0


@app.get("/generate_sample/")
def generate_sample(steps: int, n_images: int, seed: int = 0):
    try:
        image_grid = eval_gcs("best.ckpt", steps, n_images, seed)
    except AssertionError as message:
        response = {
            "message": "error " + str(message),
            "status-code": HTTPStatus.BAD_REQUEST,
        }
        return response

    test_dir = "gcs/butterfly_jar/current_data"
    os.makedirs(test_dir, exist_ok=True)
    save_point = test_dir+f"/image_grid_{steps}_{n_images}_{seed}.png"
    image_grid.save(save_point)
    # Create a blob from a file-like object
    blob = bucket.blob("current_data"+f"/image_grid_{steps}_{n_images}_{seed}.png")
    # Upload the file
    blob.upload_from_filename(save_point)
    return FileResponse(save_point)
