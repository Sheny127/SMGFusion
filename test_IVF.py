from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
from smg_fusion import MS_GAT_Fusion
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r"models/SMG.pth" 

for dataset_name in ["MSRS","TNO"]:
    print("\n" * 2 + "=" * 120) 
    model_name = "SMG"
    print("Processing dataset: " + dataset_name)
    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('test_result', dataset_name)
    if not os.path.exists(test_out_folder):
        os.makedirs(test_out_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(MS_GAT_Fusion(dim=64, scales=[64, 128, 256])).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        if not os.path.exists(os.path.join(test_folder, "ir")):
            print(f"Error: Dataset {dataset_name} path not found!")
            continue
            
        img_list = os.listdir(os.path.join(test_folder, "ir"))
        print(f"Found {len(img_list)} images. Starting inference...")
        
        for img_name in img_list:
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            
            feature_F_B = BaseFuseLayer(feature_I_B, feature_V_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)
            
    print(f"Finished processing {dataset_name}. Results saved in {test_out_folder}")
    print("=" * 120)
