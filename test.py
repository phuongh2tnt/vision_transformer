"""
Thanh Le  16 April 2024
How to test a trained model with a single input image
"""
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

CLASS = {0: 'Cat', 1: 'Dog'}

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

    # 3. Load an input image
    img = Image.open('dataset/cat_dog/test/dogs/dog_521.jpg')
    plt.imshow(img)
    plt.show()

    # 4. Resize and convert the image to a tensor
    img = T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR)(img)
    img = T.ToTensor()(img).to('cuda', dtype=torch.float).unsqueeze(dim=0)  # expand along the first dimension

    # 5. Perform a forward pass
    logits = model(img)

    # 6. Get the prediction
    prediction = logits.argmax(axis=1).item()
    print(f'This is a {CLASS[prediction]}')
