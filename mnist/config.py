from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path


class FlagsConfig(BaseModel):
    use_wandb: bool = True
    missing_imputation: bool = True


class PathConfig(BaseModel):
    save_model_dir: Path = r'mnist\experiments\demo'


class DataConfig(BaseModel):
    missing_data_percent: float = 0.5


class TrainingConfig(BaseModel):
    num_epochs: int = 5
    batch_size: int = 16
    device: str = 'cuda'
    lr: float = 0.002
    
    n_critic: int  = 1
    alpha1: float = 5.


class ModelConfig(BaseModel):
    hidden_dim: int = 32
    latent_dim: int = 32
    input_shape: List[int] = [1, 28, 28]
    num_classes: int = 10


class WandBConfig(BaseModel):
    project: str = "HexaGAN"
    save_model: bool = True
    train_log_interval: int = 10


class Config(BaseModel):
    flags: FlagsConfig = FlagsConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandBConfig = WandBConfig()


settings = Config()