from http import HTTPStatus

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.models.predict_model import eval_gcs

# from fastapi.templating import Jinja2Templates
# templates = Jinja2Templates(directory="templates/")


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# http://127.0.0.1:8000/generate_sample/?steps=1&n_images=1


@app.get("/generate_sample/")
def generate_sample(steps: int, n_images: int):
    try:
        save_point = eval_gcs("/gcs/model_best/best.ckpt", steps, n_images)
    except AssertionError as message:
        response = {
            "message": "error " + str(message),
            "status-code": HTTPStatus.BAD_REQUEST,
        }
        return response
    response = {
        # "input": image,
        "output": FileResponse(save_point),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
