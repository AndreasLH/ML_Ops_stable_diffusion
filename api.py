from http import HTTPStatus

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.models.predict_model import eval

# from fastapi.templating import Jinja2Templates
# templates = Jinja2Templates(directory="templates/")


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# http://127.0.0.1:8000/generate_sample/?steps=2&n_images=16


@app.get("/generate_sample/")
def generate_sample(steps: int, n_images: int):
    try:
        save_point = eval("outputs/2023-01-13/11-41-12", steps, n_images)
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
