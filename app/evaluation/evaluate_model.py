import torch
from app.evaluation.register_evaluation_metrices import register_evaluation_metric


@register_evaluation_metric("accuracy")
class AccuracyEvaluation:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, val_loader):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        correct_predicted = 0
        total_processed = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_processed += labels.size(0)
                correct_predicted += (predicted == labels).sum().item()

        accuracy = 100 * correct_predicted / total_processed
        print(f'Test Accuracy: {accuracy:.2f}%')

        return accuracy
