import torch
import timm
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy, f1, precision
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, average_precision_score

def train_model():
    """
    Train the model over a single epoch
    :return: training loss and training accuracy
    """
    train_loss = 0.0
    train_acc = 0.0
    train_f1 = 0.0
    train_precision = 0.0
    model.train()

    for (img, label) in tqdm(train_loader, ncols=80, desc='Training'):
        img, label = img.to('cuda', dtype=torch.float), label.to('cuda', dtype=torch.long)
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        prediction = logits.argmax(axis=1)
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=2).item()
        train_f1 += f1(prediction, label, average='macro', num_classes=2).item()
        train_precision += precision(prediction, label, average='macro', num_classes=2).item()

    return train_loss / len(train_loader), train_acc / len(train_loader), train_f1 / len(train_loader), train_precision / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and validation accuracy
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    val_precision = 0.0

    with torch.no_grad():
        for (img, label) in tqdm(val_loader, ncols=80, desc='Valid'):
            img, label = img.to('cuda', dtype=torch.float), label.to('cuda', dtype=torch.long)
            logits = model(img)
            loss = loss_fn(logits, label)
            valid_loss += loss.item()

            prediction = logits.argmax(axis=1)
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            val_f1 += f1(prediction, label, average='macro', num_classes=2).item()
            val_precision += precision(prediction, label, average='macro', num_classes=2).item()

    return valid_loss / len(val_loader), val_acc / len(val_loader), val_f1 / len(val_loader), val_precision / len(val_loader)


if __name__ == "__main__":
    transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    train_dataset = ImageFolder(root='/content/drive/My Drive/AI/el/datasetelearning/train', transform=transform)
    val_dataset = ImageFolder(root='/content/drive/My Drive/AI/el/datasetelearning/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    import torchvision.models as models
    model = models.googlenet(weights='IMAGENET1K_V1')
    #model = timm.create_model('hrnet_w18', pretrained=True, num_classes=2).to('cuda')
    #model.to('cuda')
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    num_classes=2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to('cuda')

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    max_acc = 0
    epochs = 100

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_precisions = []
    val_precisions = []

    for epoch in range(epochs):
        train_loss, train_acc, train_f1, train_precision = train_model()
        val_loss, val_acc, val_f1, val_precision = validate_model()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)

        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')

        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            torch.save(model.state_dict(), '/content/drive/My Drive/AI/el/checkpoints/epoch_' + str(epoch) + '_acc_{0:.4f}'.format(max_acc) + '.pt')
            max_acc = val_acc

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(range(epochs), train_accs, label='Training Accuracy')
    plt.plot(range(epochs), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(range(epochs), train_f1s, label='Training F1 Score')
    plt.plot(range(epochs), val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(range(epochs), train_precisions, label='Training Precision')
    plt.plot(range(epochs), val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs. Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/content/drive/My Drive/AI/el/checkpoints/metrics.png')
    plt.show()
