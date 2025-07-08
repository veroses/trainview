from pydantic import BaseModel

class Training_Request(BaseModel):
    optimizer :  str
    learning_rate : float
    batch_size : int
    epochs : int
    momentum : float | None=None
    beta1 : float | None=None
    beta2 : float | None=None

class Training_Status_Response(BaseModel):
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
