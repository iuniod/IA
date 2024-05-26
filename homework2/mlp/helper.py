
def plot_learning_curves(train_acc, test_acc, train_loss, test_loss, title):
    """Plot learning curves for training and test accuracy and loss."""
    epochs = range(1, len(train_acc) + 1)  # Generate a range of epochs
    plt.figure(figsize=(14, 6))  # Create a figure for the plots

    plt.subplot(1, 2, 1)  # Create a subplot for accuracy
    plt.plot(epochs, train_acc, 'bo-', label='Train Accuracy')  # Plot training accuracy
    plt.plot(epochs, test_acc, 'ro-', label='Test Accuracy')  # Plot test accuracy
    plt.title(f'{title} - Accuracy')  # Set the title for the accuracy plot
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Accuracy')  # Label the y-axis
    plt.legend()  # Add a legend

    plt.subplot(1, 2, 2)  # Create a subplot for loss
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')  # Plot training loss
    plt.plot(epochs, test_loss, 'ro-', label='Test Loss')  # Plot test loss
    plt.title(f'{title} - Loss')  # Set the title for the loss plot
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Loss')  # Label the y-axis
    plt.legend()  # Add a legend

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the plots
