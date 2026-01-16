import os
import cv2
import numpy as np
import math
import warnings
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
import sklearn.metrics as skm
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

CONFIG = {
    "MSRS": {
        "ir_path":    r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_img\MSRS\ir",  
        "vi_path":    r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_img\MSRS\vi",  
        "fused_path": r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_result\MSRS"  
    },
    "TNO": {
        "ir_path":    r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_img\TNO\ir",
        "vi_path":    r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_img\TNO\vi",
        "fused_path": r"D:\Lab\SMG-Fusion-main\SMG-Fusion-main\test_result\TNO"
    }
}
# ==========================================

class MetricCalculator:

    @staticmethod
    def _normalize(img):
        if img is None: return None
        img = np.squeeze(img)
        
        if img.max() <= 1.01 and img.max() > -0.1: 
            img = img * 255.0
        
        img = np.clip(img, 0, 255)
        return img

    @classmethod
    def EN(cls, img):
        img = cls._normalize(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        img = cls._normalize(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        img = cls._normalize(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):
        img = cls._normalize(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)
        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        return skm.mutual_info_score(np.uint8(image_F).flatten(), np.uint8(image_A).flatten()) + \
               skm.mutual_info_score(np.uint8(image_F).flatten(), np.uint8(image_B).flatten())

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        meanF, meanA, meanB = np.mean(image_F), np.mean(image_A), np.mean(image_B)
        rAF = np.sum((image_A - meanA) * (image_F - meanF)) / np.sqrt(
            np.sum((image_A - meanA) ** 2) * np.sum((image_F - meanF) ** 2) + 1e-10)
        rBF = np.sum((image_B - meanB) * (image_F - meanF)) / np.sqrt(
            np.sum((image_B - meanB) ** 2) * np.sum((image_F - meanF) ** 2) + 1e-10)
        return (rAF + rBF) / 2

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / (np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2))) + 1e-10)
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / (np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2))) + 1e-10)
        return corr1 + corr2

    # --- Qabf Implementation ---
    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        gA, aA = cls._Qabf_getArray(image_A)
        gB, aB = cls._Qabf_getArray(image_B)
        gF, aF = cls._Qabf_getArray(image_F)
        QAF = cls._Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls._Qabf_getQabf(aB, gB, aF, gF)
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / (deno + 1e-10)

    @staticmethod
    def _Qabf_getArray(img):
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @staticmethod
    def _Qabf_getQabf(aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8
        GAF = np.zeros_like(aA)
        mask1 = gA > gF
        mask2 = gA == gF
        mask3 = gA < gF
        GAF[mask1] = gF[mask1] / (gA[mask1] + 1e-10)
        GAF[mask2] = gF[mask2]
        GAF[mask3] = gA[mask3] / (gF[mask3] + 1e-10)
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    # --- VIFF Implementation ---
    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        return cls._compare_viff(image_A, image_F) + cls._compare_viff(image_B, image_F)

    @staticmethod
    def _compare_viff(ref, dist):
        sigma_nsq = 2
        eps = 1e-10
        num, den = 0.0, 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0: win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')[::2, ::2]
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12
            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0
            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0
            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den
        return 1.0 if np.isnan(vifp) else vifp

    # --- FMI Implementation ---
    @classmethod
    def FMI(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        
        def get_grad(img):
            gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(gx**2 + gy**2)
            
        feat_A = get_grad(image_A)
        feat_B = get_grad(image_B)
        feat_F = get_grad(image_F)
        return cls.MI(feat_F, feat_A, feat_B)

    # --- SSIM & MS-SSIM ---
    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)
        # ssim from skimage
        s1 = ssim(image_F, image_A, data_range=255.0)
        s2 = ssim(image_F, image_B, data_range=255.0)
        return s1 + s2

    @classmethod
    def MS_SSIM(cls, image_F, image_A, image_B):
        """Standalone MS-SSIM using PyTorch logic"""
        image_F = cls._normalize(image_F)
        image_A = cls._normalize(image_A)
        image_B = cls._normalize(image_B)

        def to_tensor(img):
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        # Fallback if too small
        if min(image_F.shape) < 160:
            return cls.SSIM(image_F, image_A, image_B)

        t_F = to_tensor(image_F)
        t_A = to_tensor(image_A)
        t_B = to_tensor(image_B)
        
        if torch.cuda.is_available():
            t_F, t_A, t_B = t_F.cuda(), t_A.cuda(), t_B.cuda()

        res_A = cls._compute_msssim_torch(t_F, t_A)
        res_B = cls._compute_msssim_torch(t_F, t_B)
        return res_A.item() + res_B.item()

    @staticmethod
    def _compute_msssim_torch(img1, img2, window_size=11, size_average=True, val_range=255.0):
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        if img1.is_cuda: weights = weights.cuda()
        levels = weights.size()[0]
        mssim = []
        mcs = []
        
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        if img1.is_cuda: window = window.cuda()

        for i in range(levels):
            mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2

            C1 = (0.01 * val_range) ** 2
            C2 = (0.03 * val_range) ** 2

            v1 = 2.0 * sigma12 + C2
            v2 = sigma1_sq + sigma2_sq + C2
            
            ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
            cs_map = v1 / v2

            if size_average:
                mssim.append(ssim_map.mean())
                mcs.append(cs_map.mean())
            else:
                mssim.append(ssim_map.mean(1).mean(1).mean(1))
                mcs.append(cs_map.mean(1).mean(1).mean(1))

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        output = torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
        return output

def read_image(path):

    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)

def find_file_ignore_ext(folder, filename_no_ext):

    for f in os.listdir(folder):
        if os.path.splitext(f)[0] == filename_no_ext:
            return f
    return None

def main():
    print("Start Independent Evaluation..." + "\n" + "="*120)
    
    for task_name, paths in CONFIG.items():
        ir_dir = paths["ir_path"]
        vi_dir = paths["vi_path"]
        fused_dir = paths["fused_path"]

        if not os.path.exists(fused_dir):
            print(f"[Skip] Fused folder not found: {fused_dir}")
            continue

        print(f"Dataset: [{task_name}]")
        print(f"  - IR: {ir_dir}")
        print(f"  - VI: {vi_dir}")
        print(f"  - Fused: {fused_dir}")

        fused_files = [f for f in os.listdir(fused_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif'))]
        if not fused_files:
            print("  [Error] No images found in fused folder.")
            continue

        metric_sum = np.zeros(12)
        count = 0

        for i, f_name in enumerate(fused_files):
            name_no_ext = os.path.splitext(f_name)[0]
            
            ir_name = find_file_ignore_ext(ir_dir, name_no_ext)
            vi_name = find_file_ignore_ext(vi_dir, name_no_ext)

            if not ir_name or not vi_name:
                continue

            fi = read_image(os.path.join(fused_dir, f_name))
            ir = read_image(os.path.join(ir_dir, ir_name))
            vi = read_image(os.path.join(vi_dir, vi_name))

            h = min(fi.shape[0], ir.shape[0], vi.shape[0])
            w = min(fi.shape[1], ir.shape[1], vi.shape[1])
            fi, ir, vi = fi[:h, :w], ir[:h, :w], vi[:h, :w]

            m = np.array([
                MetricCalculator.EN(fi),
                MetricCalculator.SD(fi),
                MetricCalculator.SF(fi),
                MetricCalculator.MI(fi, ir, vi),
                MetricCalculator.SCD(fi, ir, vi),
                MetricCalculator.VIFF(fi, ir, vi),
                MetricCalculator.Qabf(fi, ir, vi),
                MetricCalculator.SSIM(fi, ir, vi),
                MetricCalculator.CC(fi, ir, vi),
                MetricCalculator.AG(fi),
                MetricCalculator.FMI(fi, ir, vi),
                MetricCalculator.MS_SSIM(fi, ir, vi)
            ])
            
            metric_sum += m
            count += 1
            print(f"\r  Progress: {count}/{len(fused_files)}", end="")

        print("\n")
        if count > 0:
            avg = metric_sum / count
            print("-" * 120)
            print(f"Final Results for {task_name}:")
            print("\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tCC\tAG\tFMI\tMS-SSIM")
            print('\t' + '\t'.join([f"{val:.2f}" for val in avg]))
            print("-" * 120 + "\n")
        else:
            print("  [Warning] No valid image pairs processed.\n")

if __name__ == "__main__":
    main()