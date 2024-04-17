"""
Thanh Le  16 April 2024
How to measure the performance of a trained model
"""
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
from torchmetrics.functional import accuracy, precision, recall, f1_score


def eval_model():
    """
    Evaluate the model over a single epoch
    :return: classification performance, including accuracy, precision, recall, and F1 score
    """
    model.eval()
    acc = 0.0
    pre = 0.0
    rec = 0.0
    f1 = 0.0

    with torch.no_grad():
        for (img, label) in tqdm(test_loader, ncols=80, desc='Evaluation'):
            # Get a batch
            img, label = img.to('cuda', dtype=torch.float), label.to('cuda', dtype=torch.long)

            # Perform a feed-forward pass
            logits = model(img)

            # Get the predictions
            prediction = logits.argmax(axis=1)

            # Get the predictions to calculate the model's performance for every iteration
            acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            pre += precision(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            rec += recall(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            f1 += f1_score(prediction, label, task='multiclass', average='macro', num_classes=2).item()

    return acc / len(test_loader), pre / len(test_loader), rec / len(test_loader), f1 / len(test_loader)


if __name__ == "__main__":
    # 1. Create a new deep model
    import torchvision.models as models

    model = models.googlenet(weights='IMAGENET1K_V1')
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.to('cuda')

    # 2. Load the weights trained on the Cat-Dog dataset
    model.load_state_dict(torch.load('checkpoints/epoch_8_acc_0.9750.pt', 'cuda'))
    model.eval()

    # 3. Load the test dataset
    transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    test_dataset = ImageFolder(root='dataset/cat_dog/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 4. Evaluate the model on the test set
    acc, pre, rec, f1 = eval_model()
    print(f'Accuracy: {acc}, Precision: {pre}, Recall: {rec}, F1 score: {f1}')
