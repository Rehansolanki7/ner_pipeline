# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import nest_asyncio
from fastapi.openapi.utils import get_openapi
from training import train_model
from prediction import get_model, final_output, get_masked_data
from data_pipeline import get_text_and_labels

nest_asyncio.apply()

class Item(BaseModel):
    data_path: str
    model_path: str
    n_epoch: int

class Item1(BaseModel):
    data_path: str
    model_path: str
    fine_tune_model_path: str
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ner_train")
def ner_train1(item: Item):
    model_path = item.model_path
    data_path = item.data_path
    n_epoch = item.n_epoch
    
    try:
        txt, train_labels = get_text_and_labels(data_path)
        print(txt)
        print(train_labels)
        res = train_model(txt, train_labels, model_path, n_epoch)
        return JSONResponse(content=res)
    except Exception as e:
        output = {"Error": str(e)}
        return output

@app.post("/ner_prediction")
def ner_prediction1(item: Item1):
    model_path = item.model_path
    fine_tune_model_path = item.fine_tune_model_path
    text = item.text
    data_path = item.data_path
    try:
        txt, train_labels = get_text_and_labels(data_path)
        labels = {x for l in train_labels for x in l}
        labels = list(labels)
        prediction_result = final_output(model_path, fine_tune_model_path, labels, text)
        masked_result = get_masked_data(text, prediction_result)
        return JSONResponse(content=[prediction_result, masked_result])
    except Exception as e:
        output = {"Error": str(e)}
        return output

if __name__ == '__main__':
    uvicorn.run(app)
