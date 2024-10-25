from typing import List
from pydantic import BaseModel


class FlagsConfig(BaseModel):
    use_wandb: bool = False
    split: bool = False
    missing_masks: bool = False
    original_classifier: bool = False
    hexagan: bool = True
    impute: bool = True
    impute_classifier: bool = True
    evaluate_classifier: bool = True


class MnistConfig(BaseModel):
    input_shape: List[int] = [1, 28, 28]
    num_classes: int = 10
    
    seed: int = 42
    train_size: float = 0.6
    test_size: float = 0.2
    mi_size: float = 0.2
    

class WandbConfig(BaseModel):
    project: str = 'HexaGAN'


class ClassifierConfig(BaseModel):
    num_epochs: int = 5
    train_batch_size: int = 32
    test_batch_size: int = 64

    train_log_step: int = 10

    
class HexaganConfig(BaseModel):
    hidden_dim: int = 32
    
    num_epochs: int = 15
    train_batch_size: int = 32
    val_batch_size: int = 64
    
    lr: float = 0.002
    n_critic: int  = 1
    alpha1: float = 5.
    
    train_log_interval: int = 10
    visualize_interval: int = 50


class ImputeConfig(BaseModel):
    batch_size: int = 64


class Config(BaseModel):
    device: str = 'cuda'
    flags: FlagsConfig = FlagsConfig()
    wandb: WandbConfig = WandbConfig()
    mnist: MnistConfig = MnistConfig()
    cls: ClassifierConfig = ClassifierConfig()
    hexa: HexaganConfig = HexaganConfig() 
    impute: ImputeConfig = ImputeConfig()
    
    
settings = Config()
