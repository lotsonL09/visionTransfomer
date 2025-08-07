from fastapi import FastAPI,Request,File,UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os

import torch

from data_augmentation import get_frames,ViViTFactorized,encode_frame_to_base64

import matplotlib.pyplot as plt

import cv2

app=FastAPI()

app.mount("/static",StaticFiles(directory="static"),name="static")

templates=Jinja2Templates(directory="templates")

device='cuda' if torch.cuda.is_available() else 'cpu'
device

@app.get("/")
async def home(request:Request):
    context={
        'request':request
    }
    return templates.TemplateResponse("interface.html",context)

classes=["Actividad normal","Dejar","Levantar"]

ViVit_instance=ViViTFactorized(in_channels=3,num_classes=3,num_frames=10,img_size=64)

ViVit_instance.load_state_dict(torch.load(f="models/video_vision_transformer_200_epochs.pth"))

ViVit_instance.to(device)

ViVit_instance.eval()


@app.post('/send_video')
async def get_video(video:UploadFile = File(...)):
    #content = await video.read()
    
    temp_path=f"temp_{video.filename}"

    

    with open(temp_path,'wb') as buffer:
        shutil.copyfileobj(video.file,buffer)
    
    frames=get_frames(temp_path,n_frames=10)

    os.remove(temp_path)

    ########### Prediction of the model

    #adding batchsize

    frames=frames.unsqueeze(dim=0)

    with torch.inference_mode():
        prediction=ViVit_instance(frames.to(device))

    pred_label=torch.argmax(torch.softmax(prediction,dim=1),dim=1)

    model_response=classes[pred_label]

    ############


    #print(content)

    video.file.seek(0)

    temp_path_2=f"temp_{video.filename}_2"

    with open(temp_path_2,'wb') as buffer:
        shutil.copyfileobj(video.file,buffer)
    
    original_frames=get_frames(temp_path_2,n_frames=10,transform=None)

    encoder_frames=[encode_frame_to_base64(f) for f in original_frames.squeeze(dim=0)]

    os.remove(temp_path_2)

    return {'response':"Recibido",
            "model_response":model_response,
            "frames":encoder_frames}


if __name__ == "__main__":
    uvicorn.run(
        'main:app',
        host="127.0.0.1",
        port=8080,
        reload=True
    )