import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
session = ort.InferenceSession("defect_model.onnx")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["Clean", "Scratch", "Crack", "Contamination", "Other"]

img = Image.open("sample.jpg").convert("RGB")
img = transform(img).unsqueeze(0).numpy()

outputs = session.run(None, {"input": img})
pred = np.argmax(outputs[0])

print("🧠 Predicted Defect:", classes[pred])
