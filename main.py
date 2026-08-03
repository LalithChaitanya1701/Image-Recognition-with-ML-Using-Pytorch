import os
import sys
import urllib.request
import torch
from torchvision import models, transforms
from PIL import Image

CLASSES_FILE = "imagenet_classes.txt"
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

def download_classes():
    """Ensure ImageNet class labels file is available locally."""
    if not os.path.exists(CLASSES_FILE):
        print(f"Downloading ImageNet class labels from {CLASSES_URL}...")
        try:
            urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Failed to download class labels: {e}")
            sys.exit(1)

def get_image_path():
    """Determine image path from CLI argument, Colab upload, or user input."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    
    # Check if running in Google Colab environment
    try:
        from google.colab import files
        print("Please upload an image file:")
        uploaded = files.upload()
        if uploaded:
            return list(uploaded.keys())[0]
    except ImportError:
        pass

    # Fallback to interactive CLI prompt
    path = input("Enter path to image file: ").strip()
    if os.path.exists(path):
        return path
    else:
        print(f"Error: File '{path}' does not exist.")
        sys.exit(1)

def main():
    # 1. Download class labels if missing
    download_classes()
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    # 2. Define Image Preprocessing Transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Load Pre-trained ResNet-101 Model
    print("Loading pre-trained ResNet-101 model...")
    try:
        weights = models.ResNet101_Weights.DEFAULT
        resnet = models.resnet101(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        resnet = models.resnet101(pretrained=True)

    resnet.eval()

    # 4. Load and Preprocess Image
    img_path = get_image_path()
    print(f"\nProcessing image: {img_path}")
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # 5. Perform Inference
    print("Running image classification...")
    with torch.no_grad():
        out = resnet(batch_t)

    # 6. Extract Top Predictions and Probabilities
    probabilities = torch.nn.functional.softmax(out, dim=1)[0] * 100
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("\n" + "=" * 40)
    print("        CLASSIFICATION RESULTS        ")
    print("=" * 40)
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        score = top5_prob[i].item()
        print(f"{i+1}. {labels[idx]:<25} {score:6.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    main()