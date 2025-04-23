# Augmenting Training Data with Next-Gen AI Models for Improved Out-of-Distribution Image Classification 
This repository contains scripts and modules to train a Convolutional Neural Network (CNN) on a (custom) image dataset
with the aim of creating a synthetic Out-of-Distribution (OOD) test dataset using Diffusion Models (DMs). For the 
training process, different augmentation techniques can be used, ranging from AutoAugment to training-free DM inpainting 
using state-of-the-art DMs. For an optimal CNN training, grid search can be used finding the best parameter for a 
variety of parameter choices. The workflow contains a modular structure, is configuration-driven and flexible usage in 
training, testing and augmentation ensures that robustness for real-world scenarios can be achieved in the field of 
Image Recognition.

# Getting Started
1. **Clone the repository**
```bash
git clone https://github.com/MarcelSchi/SyntheticOODAugment
cd SyntheticOODAugment
```
2. **Create a Python environment** 
```bash 
conda create --name _myenv_ python=3.10
conda activate _myenv_ 
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```


# Configuration

All scripts read from a configuration file (e.g: ``config/config_training1.json``). You can adapt existing config files or
create a new JSON file, depending on your use-case:
```bash
{
    "mean": [0.485, 0.456, 0.406],  # ImageNet standard
    "std": [0.229, 0.224, 0.225],  # ImageNet standard
    "input_shape": 224,  # required sized for EfficientNetB0
    "grayscale": false,
    "number_epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "save_images": true,
    "augm_type": "base",  # or: autoaugment, albumentations, SD_XL, Kandinsky, Flux_Fill_Pro
    "augm_probability": 0,  # increase for more images to be augmented
    "evaluation_type": "f1",  # or: accuracy
    "prompt_dir": "app/preprocessing/prompts/prompts_training1.json",
    "mask_path": "app/data/mask_images/mask_1.jpg",
    "train_dir": "app/data/training_dataset",
    "val_dir": "app/data/validation_dataset",
    "test_dir": "app/data/test_dataset"
}
```


# Build and Test

1. **Standard Training & Evaluation**
```bash
python -m app.scripts.run_training_evaluation_pipeline
```
The ``run_training_evaluation_pipeline`` script will:
1. Load a training, validation and testing dataset
2. Apply base (validation & test) or augmentation (training, if augmentation on) transformations
3. Load pre-trained EfficientNetB0 (```num_classes=5``` to classify)
4. Train for ```number_epochs```
5. Evaluate on the test dataset, using ```evaluation_type```
#### Configurable Parameters: 
``mean``, ``std``, ``input_size``, ``learning_rate``, ``batch_size``, ``number_epochs``, ``grayscale``, ``save_images``, 
``augm_type``, ``augm_probability``, ``prompt_dir``, ``mask_path``, ``train_dir``, ``val_dir``, ``test_dir``, 
``evaluation_type``

2. **Generate Augmented Dataset**
```bash
python -m app.scripts.run_augmentation_on_dataset 
```

The separate augmentation function will create a new folder of augmented images, saving images in the respective 
directory for each augmentation type (e.g for SD_XL: saved images can be found in ``augmented_images/SD_XL/``).
``augm_type`` & ``augm_probability`` are applied to the dataset which is specified in the ``test_dir``. For a 
diffusion-based augmentation, masks from ``mask_path`` specify the regions for images to add objects, creating synthetic
data. Important: For this workflow, data should be divided into different classes that contain similar regions to be
augmented. 

#### Configurable Parameters: 
``mean``, ``std``, ``augm_type``,``augm_probability``, ``prompt_dir``, ``mask_path``, ``test_dir``

3. **Hyperparameter Grid Search**
```bash
python -m app.scripts.run_grid_search
```

This function uses a specified grid of parameter values and uses these arguments to run the ``run_training_evaluation_pipeline`` script over several 
repeats. Following functionality is included: 
- ```repeats``` defines the setting, how often each parameter combination is used to run the ``run_training_evaluation_pipeline.py`` script. The 
average score is more meaningful for a large number of runs, however computational complexity increases respectively.
- Each score within the repeats is saved in ``app/results/experiment_results.json``
- All models are saved within the directory ``temp_models`` while the respective best performing model within a
parameter combination is saved in the directory ``best_models`` 

#### Configurable Parameters: 
``mean``, ``std``, ``input_size``, ``learning_rate``, ``batch_size``, ``number_epochs``, ``grayscale``, ``save_images``, 
``augm_type``, ``augm_probability``, ``prompt_dir``, ``mask_path``, ``train_dir``, ``val_dir``, ``test_dir``, 
``evaluation_type``

4. **Evaluate pre-trained Model**

```bash
python -m app.scripts.evaluate_trained_model
```
 
This script loads any save .pth model and computes a score on a specified test dataset. Therefore, models do not need
to be re-trained for the usage of a new ``evaluation_type`` or ``test_dir``. This function is useful for analyzing 
generalization to new scenarios or applying different evaluation metrics.

#### Configurable Parameters: 
``mean``, ``std``, ``input_size``, ``grayscale``, ``test_dir``, ``evaluation_type``


# Inspiration for Future Studies

1. Flexibly add ned Augmentation strategies by extending the ``app/augmentation`` directory. Add ned ``augm_type``
2. Extend the usage of Image Recognition models by including additional model types, similar to the registration of 
``register_augmentation_strategies.py`` or ``register_evaluation_metrices.py``
3. Extend studies to create automatic mask generation - Depending on the dataset, different approaches may lead to new 
insights into less time-consuming mask generation processes.
4. Test on several test datasets - Creating not only synthetic OOD data, but rather use actual data that differs
fundamentally from the training data helps to study generalization performances of different Image Recognition models.
