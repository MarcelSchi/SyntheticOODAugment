import torch
import torch.optim as optim
import torch.nn as nn
from app.training.check_early_stopping import ModelEvaluator
from app.training.train_one_epoch import train_one_epoch
from app.optimization.calculate_validation_loss import calculate_validation_loss


def training_loop(model, train_loader, val_loader, conf):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.config.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    num_epochs = conf.config.number_epochs
    evaluator = ModelEvaluator(tolerance=5, min_improvement=0.005)
    
    for epoch in range(num_epochs):
        running_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss = calculate_validation_loss(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}; validation loss: {validation_loss}")

        if evaluator.check_for_early_stopping(validation_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), 'efficientnet.pth')
