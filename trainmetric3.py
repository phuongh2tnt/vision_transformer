"""
Thanh Le  16 April 2024
How to train/fine-tune a pre-trained model on a custom dataset (i.e., transfer learning)
"""
import torch
import timm
#sua code nay cho fine tune
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy
from torchvision.transforms import ToTensor, Resize
import numpy as np



def train_model():
    """
    Train the model over a single epoch
    :return: training loss and training accuracy
    """
    train_loss = 0.0
    train_acc = 0.0
    model.train()

    for (img, label) in tqdm(train_loader, ncols=80, desc='Training'):
        # Get a batch
        img, label = img.to('cuda', dtype=torch.float), label.to('cuda', dtype=torch.long)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Perform a feed-forward pass
        logits = model(img)

        # Compute the batch loss
        loss = loss_fn(logits, label)

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

        # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
        prediction = logits.argmax(axis=1)
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=2).item()

    return train_loss / len(train_loader), train_acc / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and validation accuracy
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for (img, label) in tqdm(val_loader, ncols=80, desc='Valid'):
            # Get a batch
            img, label = img.to('cuda', dtype=torch.float), label.to('cuda', dtype=torch.long)

            # Perform a feed-forward pass
            logits = model(img)

            # Compute the batch loss
            loss = loss_fn(logits, label)

            # Accumulate the batch loss
            valid_loss += loss.item()

            # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
            prediction = logits.argmax(axis=1)
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=2).item()

    return valid_loss / len(val_loader), val_acc / len(val_loader)


if __name__ == "__main__":

    # 1. Load the dataset
    transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    train_dataset = ImageFolder(root='/content/drive/My Drive/AI/el/datasetelearning/train', transform=transform)
    val_dataset = ImageFolder(root='/content/drive/My Drive/AI/el/datasetelearning/test', transform=transform)

    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # 3. Create a new deep model with pre-trained weights googlenet
    import torchvision.models as models
    model = models.googlenet(weights='IMAGENET1K_V1')
    #3.1. Create a new deep model use timm
    #model=timm.create_model('hrnet_w18', pretrained=True, num_classes=2).to('cuda')

    # 4. Note that the model pre-trained model has 1,000 output neurons (because ImageNet has 1,000 classes), so we must
    # customize the last linear layer to adapt to our 2-class problem (i.e., Cat vs Dog)
    #cua torchvision cu
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    num_classes=2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to('cuda')

    # 4. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    #ham mat mat moi theo kieu timm
    #loss_fn = nn.CrossEntropyLoss()
    # 5. Train the model with 100 epochs
    max_acc = 0
    acc_trains=[]
    acc_vals=[]
    loss_trains=[]
    loss_vals=[]
    epochs=3
    for epoch in range(epochs):

        # 5.1. Train the model over a single epoch
        train_loss, train_acc = train_model()

        # 5.2. Validate the model after training
        val_loss, val_acc = validate_model()

        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')
        acc_trains.append((epoch, train_acc))
        acc_vals.append((epoch, val_acc))
        loss_trains.append((epoch, train_loss))
        loss_vals.append((epoch, val_loss))
        # 4.3. Save the model if the validation accuracy is increasing
        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            torch.save(model.state_dict(),'/content/drive/My Drive/AI/el/checkpoints/2epoch_' + str(epoch) + '_acc_{0:.4f}'.format(max_acc) + '.pt')
            max_acc = val_acc
    
    #in mang
    train_accs=np.array(acc_trains)
    val_accs=np.array(acc_vals)
    train_losss=np.array(loss_trains)
    val_losss=np.array(loss_vals)
    second = [row[1] for row in train_accs]
    print(second)
    print("Ve mo hinh")
    x = [row[0] for row in train_accs]
    y = [row[1] for row in train_accs]

    # Plot
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Data')
    plt.grid(True)
    plt.show()
    # Plotting
    plt.figure(figsize=(12, 6))

    #plt.subplot(2, 2, 1)
    plt.plot(range(epochs), train_accs, label='Training Accuracy')
    plt.plot(range(epochs), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    #plt.subplot(2, 2, 2)
    plt.plot(range(epochs), train_losss, label='Training Loss')
    plt.plot(range(epochs), val_losss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
