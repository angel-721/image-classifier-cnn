import torch
from torchvision import transforms
from PIL import Image
from torch import nn 
from train import CNN

def predict(image_path="./inputs/test.png", model_path="./models/best.pth"):
    network = CNN("./data")
    network.load_state_dict(torch.load(model_path))
    image = Image.open(image_path).convert('RGB')
    data_transforms = transforms.Compose(
        [
            transforms.Resize((53,53)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    transformed_image = data_transforms(image)
    with torch.no_grad():
        output = network(transformed_image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        predicted_class = network.classes[predicted]
    print("Predicted Class: ", predicted_class)

print(predict())