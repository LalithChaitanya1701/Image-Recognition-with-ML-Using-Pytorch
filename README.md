# Image Recognition using PyTorch & ResNet-101

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An image recognition pipeline built in Python using PyTorch and Torchvision. The application utilizes a pre-trained **ResNet-101** deep neural network to perform image classification over the 1,000 ImageNet categories.

---

## Features

- **Pre-trained Deep Learning**: Employs PyTorch's ResNet-101 model trained on the ImageNet dataset.
- **Cross-Platform**: Supports running via Command Line Interface (CLI), interactive Python scripts, and Google Colab.
- **Automatic Class Label Fetching**: Automatically downloads `imagenet_classes.txt` if not present.
- **Top-5 Predictions**: Returns the top 5 predicted categories along with confidence percentages.
- **Standard Preprocessing**: Uses standard ImageNet transformations (Resize, CenterCrop, Normalization).

---

## Repository Structure

```text
Image_Recog_ML_Pytorch/
├── main.py                  # Main inference script
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules for Python/PyTorch/IDEs
├── LICENSE                  # MIT License file
└── README.md                # Project documentation
```

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Image_Recog_ML_Pytorch.git
   cd Image_Recog_ML_Pytorch
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Command Line (CLI)
Pass the path to an image as a command-line argument:
```bash
python main.py path/to/your/image.jpg
```

### Interactive Mode
If no file argument is provided, the script will prompt you for an image file path:
```bash
python main.py
```

### Google Colab
If executed inside a Google Colab notebook, `main.py` automatically opens an upload dialog to select an image from your local computer.

---

## Sample Output

```text
Loading pre-trained ResNet-101 model...

Processing image: sample.jpg
Running image classification...

========================================
        CLASSIFICATION RESULTS        
========================================
1. golden retriever            94.32%
2. Labrador retriever           4.15%
3. Kuvasz                       0.82%
4. Chesapeake Bay retriever     0.31%
5. Tennis ball                  0.10%
========================================
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
