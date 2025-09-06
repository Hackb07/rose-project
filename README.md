

#  Rose Classification using ResNet18

This project implements a **deep learning-based image classification system** for detecting and categorizing rose images using **Transfer Learning with ResNet18**.
It leverages **PyTorch** for training and evaluation, with advanced data augmentations and multiple model export formats for deployment (PyTorch, TorchScript, ONNX).

---

##  Project Structure

```
rose-project/
│── DATASET/                # Dataset (train/valid/test directories)
│── SavedModels/           # Folder where trained models & metrics are saved
│   ├── RoseResNet18_cpu.pth   # Best model weights (PyTorch)
│   ├── RoseResNet18_cpu.pt    # TorchScript model
│   ├── RoseResNet18_cpu.onnx  # ONNX export
│   ├── train_losses.npy       # Training loss history
│   ├── val_losses.npy         # Validation loss history
│   ├── train_accs.npy         # Training accuracy history
│   ├── val_accs.npy           # Validation accuracy history
│   ├── loss_curve.png         # Loss graph
│   ├── accuracy_curve.png     # Accuracy graph
│── train.ipynb                # Main training script
│── README.md               # Project documentation
```

---

##  Features

* **Transfer Learning with ResNet18** (pretrained on ImageNet)
* **Strong Data Augmentations** (rotation, flips, color jitter, affine transforms)
* **Multi-device Support** (GPU / CPU automatically detected)
* **Automatic Model Saving** (PyTorch `.pth`, TorchScript `.pt`, ONNX `.onnx`)
* **Learning Rate Scheduler** for stable convergence
* **Training & Validation Metrics Visualization**

---

##  Dataset

The dataset is organized into:

```
DATASET/
│── train/        # Training images
│── valid/        # Validation images
│── test/         # Test images
```

Each subfolder should contain class-wise directories (e.g., `rose-red`, `rose-white`, etc.).

---

##  Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Hackb07/rose-project.git
   cd rose-project
   ```

2. Install dependencies:

   ```bash
   pip install torch torchvision matplotlib numpy
   ```

3. Verify dataset structure inside `rose-3/`.

---

##  Training

Run the notebook :

```bash
train.ipynb
```

During training:

* Models are evaluated on the validation set.
* Best model is saved automatically in `SavedModels/`.
* Training progress is logged with loss & accuracy curves.

---

##  Results

* The best model achieves **Validation Accuracy = \~X%** (replace with your achieved accuracy).
* Training & Validation Loss/Accuracy curves are saved in:

  * `SavedModels/loss_curve.png`
  * `SavedModels/accuracy_curve.png`

---

##  Model Exports

Each trained model is exported in **three formats** for deployment:

* **PyTorch State Dict (`.pth`)** → for retraining/fine-tuning.
* **TorchScript (`.pt`)** → for mobile & edge device inference.
* **ONNX (`.onnx`)** → for cross-platform deployment (TensorRT, OpenVINO, etc.).

---

## Testing

Evaluate the trained model on the test set:

```python
# Inside test.py (you can create it using test_loader)
cnn_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = cnn_model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.4f}")
```

---

##  Future Improvements

* Fine-tune earlier layers of ResNet for better accuracy.
* Experiment with **ResNet50, EfficientNet, or Vision Transformers**.
* Implement **Grad-CAM visualizations** for interpretability.
* Deploy as a **Flask/Django Web App** or **Mobile App**.

---

##  Requirements

* Python 3.8+
* PyTorch >= 1.9
* torchvision
* matplotlib
* numpy

---

##  Author

**B. Tharun Bala**
*Artificial Intelligence & Data Science Student*
Roll No: 610823U243059
College: Perumal Manimekalai College of Engineering, Hosur

---
