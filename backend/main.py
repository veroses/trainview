import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import Training_Request, Training_Status_Response
from .training import train_model
from threading import Event, Thread
import time

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

training_thread = None

stop_event = Event()

@app.get("/")
async def root():
    return {"message" : "API working"}

@app.post("/start-training")
async def start_training(params : Training_Request, background_tasks : BackgroundTasks):
    global training_thread
    global stop_training

    if training_thread and training_thread.is_alive():
        print("Stopping previous training...")
        stop_event.set()
        training_thread.join()
        print("Previous training stopped.")
        time.sleep(1)

    stop_event = Event()
    training_status.epoch = 0
    training_status.loss = 1
    training_status.accuracy = 0

    training_thread = Thread(target=train_model, args=(params, training_status, stop_event))
    training_thread.start()

    return {"message" : "training started successfully"}

@app.get("/training-status", response_model=Training_Status_Response)
async def get_training_status():
    return training_status

@app.get("/ping")
async def ping():
    return {"message" : "pong"}

