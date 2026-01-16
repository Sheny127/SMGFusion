# SMG-Fusion: A Superpixel-guided Mixture of Experts Graph Network for Multi-Modality Image Fusion

This is the official PyTorch implementation of the paper **"SMG-Fusion: A Superpixel-guided Mixture of Experts Graph Network for Multi-Modality Image Fusion"**.

## 📂 Project Structure

The directory structure for training and testing should be organized as follows:

```text
SMG-Fusion/
├── data/                  # Contains pre-processed .h5 training data (e.g., MSRS)
├── models/                # Pre-trained SMG-Fusion weights
├── weights/               # Pre-trained YOLOv9-c weights for detection
├── test_img/              # Source images for inference
│   ├── MSRS/              # Dataset Name
│   │   ├── ir/            # Infrared images
│   │   └── vi/            # Visible images
│   ├── TNO/
│   │   ├── ir/
│   │   └── vi/
│   └── ...
├── test_result/           # Output folder for fused images
├── dataprocessing.py      # Script to convert raw images to .h5 format
├── train.py               # Script for training the model
├── test_IVF.py            # Main inference script
├── RGB.py                 # Tool to restore color from grayscale fusion results
├── smg_fusion.py          # Network architecture definition
└── requirements.txt       # Python dependencies
```

## 🛠️ Environment Setup

Please install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Data Preparation

We provide the pre-processed **MSRS** dataset in the `data/` folder, so you can start training immediately.

If you want to train on a custom dataset, organize your images and run the processing script to convert them into `.h5` format:

```bash
python dataprocessing.py
```

### 2. Training

To train the SMG-Fusion model from scratch, run:

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

### 4. Color Restoration

Since the network processes images in grayscale to focus on structure and texture, the raw output might be single-channel. To restore the color information from the original visible image (YCbCr conversion), run:

```bash
python RGB.py
```

## 📚 Datasets

We evaluated our method on the following public datasets. Please cite the original authors if you use these datasets.

*   **MSRS Dataset**: [[Link to Repository](https://github.com/Linfeng-Tang/MSRS)]
    *   Used for training and testing. Contains aligned IR-VIS pairs with semantic labels.
*   **RoadScene Dataset**: [Link to Repository](https://github.com/hanna-xu/RoadScene)]
    *   Focuses on road traffic scenarios with complex illumination.
*   **TNO Dataset**: [Link to Dataset](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)]
    *   Classic benchmark for military and surveillance scenarios.
*   **M3FD Dataset**: [Link to Repository](https://github.com/JinyuanLiu-CV/TarDAL)]
    *   Used for the downstream object detection task.

## 📧 Contact

If you have any questions, please contact: `2408540010@kmu.stu.edu.cn`
