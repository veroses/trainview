import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import Training_Request, Training_Status_Response
from .training import train_model


app = FastAPI()

origins  = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_status = Training_Status_Response()

@app.get("/")
async def root():
    return {"message" : "API working"}

@app.post("/start-training")
async def start_training(params : Training_Request, background_tasks : BackgroundTasks):
    background_tasks.add_task(train_model, params, training_status)
    return {"message" : "training started successfully"}

@app.get("/training-status", response_model=Training_Status_Response)
async def get_training_status():
    return training_status

@app.get("/ping")
async def ping():
    return {"message" : "pong"}

