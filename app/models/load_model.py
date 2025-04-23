from app.models.efficientnet import EfficientNet
import torch


def load_model(num_classes=5, grayscale=False):
    model = EfficientNet(num_classes=num_classes, grayscale=grayscale)

    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
