# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia
from smg_fusion import MS_GAT_Fusion
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criteria_fusion = Fusionloss()
model_str = 'SMG'

# --- Hyper-parameters ---
num_epochs = 120 
epoch_gap = 40

lr = 1e-4
weight_decay = 0
batch_size = 8 

# Coefficients
coeff_mse_loss_VF = 1. 
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      
coeff_tv = 5.

coeff_grad = 20.0 
coeff_int = 5.0 

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# --- Model ---
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(MS_GAT_Fusion(dim=64, scales=[64, 128, 256])).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# --- Optimizer ---
optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

# --- Scheduler ---
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

# --- Losses ---
MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

# --- Data ---
dataset_path = r"data/MSRS_train_imgsize_128_stride_200.h5"
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")

trainloader = DataLoader(H5Dataset(dataset_path),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

if not os.path.exists("models"):
    os.makedirs("models")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

print(f"Start training SMG-Fusion on {device}...")

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # Phase I: Decomposition Training
        if epoch < epoch_gap: 
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.spatial_gradient(data_VIS),
                                   kornia.filters.spatial_gradient(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        
        else: 
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            
            # Forward
            feature_F_B = BaseFuseLayer(feature_I_B, feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)  

            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  

            grad_vis = kornia.filters.spatial_gradient(data_VIS)
            grad_ir = kornia.filters.spatial_gradient(data_IR)
            grad_fuse = kornia.filters.spatial_gradient(data_Fuse)
            
            mag_vis = torch.abs(grad_vis).sum(dim=2)
            mag_ir = torch.abs(grad_ir).sum(dim=2)
            mag_fuse = torch.abs(grad_fuse).sum(dim=2)
            
            target_grad = torch.max(mag_vis, mag_ir)
            loss_grad = L1Loss(mag_fuse, target_grad)
            
            target_pixel = torch.max(data_VIS, data_IR)
            loss_pixel = L1Loss(data_Fuse, target_pixel)

            coeff_grad = 8.0  
            
            coeff_pixel = 2.0 
            
            loss = fusionloss + \
                   coeff_decomp * loss_decomp + \
                   coeff_grad * loss_grad + \
                   coeff_pixel * loss_pixel
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        # Logging
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f] ETA: %s"
            % (epoch, num_epochs, i, len(loader['train']), loss.item(), str(time_left).split(".")[0])
        )

    # Scheduler update
    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    # LR Guard
    for opt in [optimizer1, optimizer2, optimizer3, optimizer4]:
        for param_group in opt.param_groups:
            if param_group['lr'] <= 1e-6:
                param_group['lr'] = 1e-6
    
# Save
save_path = os.path.join("models", "" + timestamp + '.pth')
torch.save({
    'DIDF_Encoder': DIDF_Encoder.state_dict(),
    'DIDF_Decoder': DIDF_Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}, save_path)
print(f"\nModel saved to {save_path}")
