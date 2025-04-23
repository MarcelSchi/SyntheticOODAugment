import matplotlib.pyplot as plt


def plot_loss_curves(training_losses, validation_losses, save_path='plots_loss_curves'):

    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r-o', label='Validation Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()
