import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os

MODEL_PATH = "model.pth"

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"'{model_path}' not found. Run main.py first.")
    checkpoint  = torch.load(model_path, map_location=torch.device("cpu"))
    class_names = checkpoint["class_names"]
    num_classes  = checkpoint["num_classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict(image_path, model, class_names):
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    predicted_class = class_names[pred_idx.item()]
    confidence      = conf.item()
    all_probs       = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}
    return predicted_class, confidence, all_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        exit(1)

    model, class_names = load_model(MODEL_PATH)
    predicted_class, confidence, all_probs = predict(args.image, model, class_names)

    print(f"\nPredicted : {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nTop 3:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {cls:<25} {prob*100:5.1f}%  {'█' * int(prob*20)}")