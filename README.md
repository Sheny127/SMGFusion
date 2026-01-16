# SMG-Fusion: A Superpixel-guided Mixture of Experts Graph Network for Multi-Modality Image Fusion

This is the official PyTorch implementation of the paper **"SMG-Fusion: A Superpixel-guided Mixture of Experts Graph Network for Multi-Modality Image Fusion"**.

## 📂 Project Structure

The directory structure is organized as follows:

```text
SMG-Fusion/
├── models/                # Pre-trained SMG-Fusion weights (.pth)
├── test_img/              # Source images for inference
│   ├── MSRS/              # Dataset Name
│   │   ├── ir/            # Infrared images
│   │   └── vi/            # Visible images
│   └── ...
├── test_result/           # Output folder for fused images
├── utils/                 # Utility scripts (image I/O, logger, etc.)
├── dataprocessing.py      # Script to convert raw images to .h5 format for training
├── eval.py                # Script for quantitative evaluation (Compute EN, SD, SSIM, etc.)
├── net.py                 # Backbone network definitions (Encoder/Decoder)
├── RGB.py                 # Tool to restore color from grayscale fusion results
├── smg_fusion.py          # Core fusion network architecture (MS_GAT_Fusion)
├── test_IVF.py            # Main inference script
└── train.py               # Script for training the model
```

## 🛠️ Environment Setup

Please ensure you have Python and PyTorch installed. Install the required dependencies:

```bash
pip install torch torchvision opencv-python numpy h5py scipy
```

## 🚀 Usage

### 1. Data Preparation

If you want to train on a custom dataset, organize your images and run the processing script to convert them into `.h5` format:

```bash
python dataprocessing.py
```

### 2. Training

To train the SMG-Fusion model from scratch:

```bash
python train.py
```
*The training configurations (epochs, batch size, learning rate) can be modified inside `train.py`.*

### 3. Inference (Testing)

To fuse infrared and visible images:

1.  Place your source images in the `test_img` directory following this structure:
    *   Infrared: `test_img/[Dataset_Name]/ir/`
    *   Visible: `test_img/[Dataset_Name]/vi/`
2.  Open `test_IVF.py` and modify the dataset name/path variable to match your target folder (e.g., `'TNO'` or `'MSRS'`).
3.  Run the inference script:

```bash
python test_IVF.py
```
The fused images will be saved in the `test_result/` folder.

### 4. Evaluation

To calculate quantitative metrics (such as EN, SD, SF, SSIM, etc.) for the fused images:

```bash
python eval.py
```

### 5. Color Restoration

Since the network processes images in grayscale to focus on structure and texture, the raw output might be single-channel. To restore the color information from the original visible image (YCbCr conversion), run:

```bash
python RGB.py
```

## 📊 Performance

Quantitative evaluation results on the **TNO** dataset:

| Model | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM | CC | AG | FMI | MS-SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SMG** | 7.10 | 44.53 | 13.57 | 2.68 | 1.63 | 0.83 | 0.61 | 1.31 | 0.49 | 5.01 | 1.51 | 1.36 |

## 📚 Datasets

We evaluated our method on the following public datasets:

*   **MSRS Dataset**: [[Link](https://github.com/Linfeng-Tang/MSRS)] - Used for training and testing.
*   **RoadScene Dataset**: [[Link](https://github.com/hanna-xu/RoadScene)]
*   **TNO Dataset**: [[Link](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)]

## 📧 Contact

If you have any questions, please contact: `2408540010@kmu.stu.edu.cn`
```
