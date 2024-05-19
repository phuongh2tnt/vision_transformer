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
from torchmetrics.functional import precision, f1_score, recall
from torchvision.transforms import ToTensor, Resize
import numpy as np



def train_model():
    """
    Train the model over a single epoch
    :return: training loss and training accuracy
    """
    train_loss = 0.0
    train_acc = 0.0
    train_pre=0.0
    train_rec=0.0
    train_f1=0.0
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
        train_pre+=precision(prediction, label, task='multiclass', average='macro', num_classes=2).item()
        train_rec+=recall(prediction, label, task='multiclass', average='macro', num_classes=2).item()
        train_f1+=f1_score(prediction, label, task='multiclass', average='macro', num_classes=2).item()

    return train_loss / len(train_loader), train_acc / len(train_loader),train_pre / len(train_loader),train_rec / len(train_loader), train_f1 / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and validation accuracy
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0
    val_pre=0.0
    val_rec=0.0
    val_f1=0.0
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
            val_pre+=precision(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            val_rec+=recall(prediction, label, task='multiclass', average='macro', num_classes=2).item()
            val_f1+=f1_score(prediction, label, task='multiclass', average='macro', num_classes=2).item()
    return valid_loss / len(val_loader), val_acc / len(val_loader),val_pre / len(val_loader),val_rec / len(val_loader), val_f1 / len(val_loader)


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
    #model = models.googlenet(weights='IMAGENET1K_V1')
    #3.1. Create a new deep model use timm
    model=timm.create_model('hrnet_w18', pretrained=True, num_classes=2).to('cuda')

    # 4. Note that the model pre-trained model has 1,000 output neurons (because ImageNet has 1,000 classes), so we must
    # customize the last linear layer to adapt to our 2-class problem (i.e., Cat vs Dog)
    #cua torchvision cu
    #num_features = model.fc.in_features
    #model.fc = torch.nn.Linear(num_features, 2)
    #num_classes=2
    #num_features = model.fc.in_features
    #model.fc = nn.Linear(num_features, num_classes)
    model.to('cuda')

    # 4. Specify loss function and optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    #loss_fn = torch.nn.CrossEntropyLoss()
    #ham mat mat moi theo kieu timm
    loss_fn = nn.CrossEntropyLoss()
    # 5. Train the model with 100 epochs
    max_acc = 0
    acc_trains=[]
    acc_vals=[]
    loss_trains=[]
    loss_vals=[]
   #me tric moi
    pre_trains=[]
    pre_vals=[]
    rec_trains=[]
    rec_vals=[]
    f1_trains=[]
    f1_vals=[]
    #khởi tạo metric moi
    train_accs=[]
    val_accs=[]
    train_losss=[]
    val_losss=[]
    train_pres=[]
    val_pres=[]
    train_recs=[]
    val_recs=[]
    train_f1s=[]
    val_f1s=[] 
    
    epochs=100

    
    for epoch in range(epochs):

        # 5.1. Train the model over a single epoch
        #train_loss, train_acc = train_model()
        #Tinh them metric
        train_loss, train_acc,train_pre,train_rec, train_f1 = train_model()

        # 5.2. Validate the model after training
        #val_loss, val_acc = validate_model()
        #Tinh them metric
        val_loss, val_acc,val_pre,val_rec, val_f1 = validate_model()

        print(f'Epoch {epoch}: Validation loss = {val_loss}, Validation accuracy: {val_acc}')
        acc_trains.append((epoch, train_acc))
        acc_vals.append((epoch, val_acc))
        loss_trains.append((epoch, train_loss))
        loss_vals.append((epoch, val_loss))
        pre_trains.append((epoch, train_pre))
        pre_vals.append((epoch, val_pre))
        rec_trains.append((epoch, train_rec))
        rec_vals.append((epoch, val_rec))
        f1_trains.append((epoch, train_f1))
        f1_vals.append((epoch, val_f1))
        # 4.3. Save the model if the validation accuracy is increasing
        if val_acc > max_acc:
            print(f'Validation accuracy increased ({max_acc} --> {val_acc}). Model saved')
            torch.save(model.state_dict(),'/content/drive/My Drive/AI/el/hrcheckpoints/2epoch_' + str(epoch) + '_acc_{0:.4f}'.format(max_acc) + '.pt')
            max_acc = val_acc
    
    #in mang
    train_accs=np.array(acc_trains)
    val_accs=np.array(acc_vals)
    train_losss=np.array(loss_trains)
    val_losss=np.array(loss_vals)
    train_pres=np.array(pre_trains)
    val_pres=np.array(pre_vals)
    train_recs=np.array(rec_trains)
    val_recs=np.array(rec_vals)
    train_f1s=np.array(f1_trains)
    val_f1s=np.array(f1_vals)     
    second = [row[1] for row in train_accs]
    print(second)
    print("Ve mo hinh train_accs")
    x = [row[0] for row in train_accs]
    y = [row[1] for row in train_accs]
    acctrain_tb=sum(y)/len(y)
    print(f'Do chinh xac huan luyen={acctrain_tb}')
    
    # Plot
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/train_acc.png')
    plt.show()
    # Plotting
    print("Ve mo hinh val_accs")
    x1 = [row[0] for row in val_accs]
    y1 = [row[1] for row in val_accs]
    accval_tb=sum(y1)/len(y1)
    print(f'Do chinh xac validation={accval_tb}')
    # Plot
    plt.plot(x1, y1, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/val_acc.png')
    plt.show()
    # Plotting
    plt.figure(figsize=(12, 6))
    print("Ve mo hinh train_loss")
    #---------ve hai do thi --------------
    plt.subplot(2, 2, 2)
    plt.plot(x, y, marker='o', linestyle='-',color='g', label='Training accuracy')
    plt.plot(x1, y1, marker='+', linestyle='-',color='b',label='Validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/acc.png')
    plt.show()
    #---------------------------------------------
    x2 = [row[0] for row in train_losss]
    y2 = [row[1] for row in train_losss]
    losstrain_tb=sum(y2)/len(y2)
    print(f'Do mat mat training={losstrain_tb}')
    # Plot
    plt.plot(x2, y2, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/train_loss.png')
    plt.show()
    print("Ve mo hinh val_loss")
    x3 = [row[0] for row in val_losss]
    y3 = [row[1] for row in val_losss]
    lossval_tb=sum(y3)/len(y3)
    print(f'Do mat mat validation={lossval_tb}')
     #---------ve hai do thi --------------
    plt.subplot(2, 2, 2)
    plt.plot(x2, y2, marker='o', linestyle='-',color='g', label='Training loss')
    plt.plot(x3, y3, marker='+', linestyle='-',color='b',label='Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/loss.png')
    plt.show()
    #---------------------------------------------
    # Plot
    plt.plot(x3, y3, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/val_loss.png')
    plt.show()
    
   #-----------------------metric moi-----------
    x4 = [row[0] for row in train_pres]
    y4 = [row[1] for row in train_pres]
    pretrain_tb=sum(y4)/len(y4)
    print(f'Precision train={pretrain_tb}')
    # Plot
    plt.plot(x4, y4, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Training Precision')
    plt.title('Training Precision vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/train_pre.png')
    plt.show()

    x5 = [row[0] for row in val_pres]
    y5 = [row[1] for row in val_pres]
    preval_tb=sum(y5)/len(y5)
    print(f'Precision validation=={preval_tb}')
    # Plot
    plt.plot(x5, y5, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Validation Precision')
    plt.title('Validation Precision vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/val_pre.png')
    plt.show()
      #-----------------
      #---------ve hai do thi --------------
    plt.subplot(2, 2, 2)
    plt.plot(x4, y4, marker='o', linestyle='-',color='g', label='Training precision')
    plt.plot(x5, y5, marker='+', linestyle='-',color='b',label='Validation precision')
    plt.xlabel('epoch')
    plt.ylabel('Precision')
    plt.title('Precision vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/pre.png')
    plt.show()
    #---------------------------------------------
    x6 = [row[0] for row in train_recs]
    y6 = [row[1] for row in train_recs]
    rectrain_tb=sum(y6)/len(y6)
    print(f'Recall train={rectrain_tb}')
    # Plot
    plt.plot(x6, y6, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Training recall')
    plt.title('Training recall vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/rec_train.png')
    plt.show()

    x7 = [row[0] for row in val_recs]
    y7 = [row[1] for row in val_recs]
    recval_tb=sum(y7)/len(y7)
    print(f'Recall train={recval_tb}')
    # Plot
    plt.plot(x7, y7, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Validation recall')
    plt.title('Validation recall vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/rec_val.png')
    plt.show()
       #-----------------
    #---------ve hai do thi --------------
    plt.subplot(2, 2, 2)
    plt.plot(x6, y6, marker='o', linestyle='-',color='g', label='Training recall')
    plt.plot(x7, y7, marker='+', linestyle='-',color='b',label='Validation recall')
    plt.xlabel('epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/rec.png')
    plt.show()
    #---------------------------------------------
    x8 = [row[0] for row in train_f1s]
    y8 = [row[1] for row in train_f1s]
    trainf1_tb=sum(y8)/len(y8)
    print(f'train f1_score={trainf1_tb}')
    # Plot
    plt.plot(x8, y8, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Training f1_score')
    plt.title('Training f1_score vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/f1score_train.png')
    plt.show()

    x9 = [row[0] for row in val_f1s]
    y9 = [row[1] for row in val_f1s]
    
    valf1_tb=sum(y9)/len(y9)
    print(f'val f1_score={valf1_tb}')
    # Plot
    plt.plot(x9, y9, marker='o', linestyle='-')
    plt.xlabel('epoch')
    plt.ylabel('Validation f1_socre')
    plt.title('Validation f1_socre vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/f1_val.png')
    plt.show()
    #plt.savefig('/content/drive/My Drive/AI/el/checkpoints/trainacc2.png')
    #---------ve hai do thi --------------
    plt.subplot(2, 2, 2)
    plt.plot(x4, y4, marker='o', linestyle='-',color='g', label='Training f1_score')
    plt.plot(x5, y5, marker='+', linestyle='-',color='b',label='Validation f1_score')
    plt.xlabel('epoch')
    plt.ylabel('f1_score')
    plt.title('f1_score vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/hrcheckpoints/f1.png')
    plt.show()
    #---------------------------------------------
    #plt.show()
    
    #plt.subplot(2, 2, 2)
    #plt.plot(train_losss, 'g',label='Training Loss')
    #plt.plot(val_losss, 'b',label='Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('/content/drive/My Drive/AI/el/checkpoints/trainacc3.png')
    plt.show()
