import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16)

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, len(test_data.classes))
model.load_state_dict(torch.load("defect_model.pth"))
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Test Accuracy: {accuracy:.2f}%")
