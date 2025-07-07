from pydantic import BaseModel

@BaseModel
class Training_Request:
    optimizer :  str
    learning_rate : float
    batch_size : int
    epochs : int
    momentum : float | None=None