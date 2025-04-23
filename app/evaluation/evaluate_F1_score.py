import torch
from sklearn.metrics import f1_score
from app.evaluation.register_evaluation_metrices import register_evaluation_metric


@register_evaluation_metric("f1")
class F1Evaluation:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, val_loader):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        f1_final_score = f1_score(all_labels, all_predictions, average='weighted')

        print(f'F1 Score (Weighted): {f1_final_score:.2f}')
        return f1_final_score
