from pydantic import BaseModel
from typing import List


class DataConfig(BaseModel):
    missing_data_percent: float = 0.5


class TrainingConfig(BaseModel):
    batch_size: int = 8


class ModelConfig(BaseModel):
    hidden_dim: int = 32
    latent_dim: int = 32
    input_shape: List[int] = [1, 28, 28]
    num_classes: int = 10


class Config(BaseModel):
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()


settings = Config()