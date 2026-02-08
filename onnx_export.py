import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(1280, 5)  # change 5 to your number of classes
model.load_state_dict(torch.load("defect_model.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "defect_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ ONNX model exported successfully")
