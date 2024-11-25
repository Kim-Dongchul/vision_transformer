import matplotlib.pyplot as plt

def show_plot(title, train_losses, test_losses):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(train_losses, label='Train', color='blue')
    plt.plot(test_losses, label='Test', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend()
    plt.show()