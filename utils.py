from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def data_transformation(transformation_config:dict):
    # Train data transformations
    train_transforms = transforms.Compose(transformation_config['train_config'])

    # Test data transformations
    test_transforms = transforms.Compose(transformation_config['test_config'])
    return train_transforms, test_transforms


def plot_dataset(train_loader):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def GetCorrectPredCount(pPrediction, pLabels):
    # Calculate the number of correct predictions
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def plot_train(train_acc:list, train_losses:list, test_acc:list, test_losses:list):
    # Plot the training and test metrics
    train_losses = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")