from app.models.load_model import load_model
import torch


# already trained model, that is saved as .pth file, can be loaded to perform new evaluation
def load_trained_model(path, num_classes=5, grayscale=False):
    model = load_model(num_classes=num_classes, grayscale=grayscale)
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
