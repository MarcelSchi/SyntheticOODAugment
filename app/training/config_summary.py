from pydantic import BaseModel


class Config_Summary(BaseModel):
    std: list
    mean: list
    input_shape: int
    number_epochs: int
    batch_size: int
    learning_rate: float
    grayscale: bool
    save_images: bool
    augm_type: str
    augm_probability: float
    evaluation_type: str
    prompt_dir: str
    mask_path: str
    train_dir: str
    val_dir: str
    test_dir: str
