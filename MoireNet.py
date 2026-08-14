"""
MoireNet-Triple - Трёхдоменная нейросеть для локализации муара
======================================================================
Архитектура объединяет три параллельных домена обработки:
    1. CNN домен - пространственные признаки
    2. DCT домен - частотные признаки
    3. Wavelet домен - пространственно-частотные признаки

Выход сети: карта границ муара (1 канал, значения в [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
import torch_dct as dct
from swt_module import SWTForward
import matplotlib.pyplot as plt

# =============================================================================
# 1. БАЗОВЫЕ КОМПОНЕНТЫ
# =============================================================================

class BasicConv(nn.Module):
    """Conv2d + SELU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.activation = nn.SELU()

    def forward(self, x):
        return self.activation(self.conv(x))


class simam_module(torch.nn.Module):
    """SimAM: механизм внимания без обучаемых параметров"""
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


# =============================================================================
# 2. ФУНКЦИИ ПОТЕРЬ
# =============================================================================
# L = w1 * L_pixel + w2 * L_dir + w3 * L_dis
#
# L_pixel - пиксельная ошибка (Smooth L1)
# L_dir - ошибка направления
# L_dis - ошибка распределения
# =============================================================================

def get_start_point(kernel_size, start_id):
    """Вычисление начальной точки для рисования линии в бинарном ядре"""
    x = max(start_id - (kernel_size - 1), 0)
    y = min(start_id, kernel_size - 1)
    return int(x), int(y)


def generate_filters(kernel_size=7, filters_num=14):
    """Генерация 14 бинарных ядер 7×7 для Direction Loss"""
    filters_num = min(2 * kernel_size, filters_num)
    sep = 2 * kernel_size / filters_num
    filters = []
    for i in range(filters_num):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        start_id = np.round(sep * i)
        x, y = get_start_point(kernel_size, start_id)
        kernel = cv2.line(kernel, (x, y), (kernel_size - x - 1, kernel_size - y - 1), 1, 1)
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        filters.append(kernel)
    return filters


class DirectionLoss(nn.Module):
    """Ошибка направления"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        kernels = generate_filters(kernel_size)
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False)
            for k in kernels
        ])
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred, target):
        loss = 0
        for kernel in self.kernels:
            kernel = kernel.to(pred.device)
            pred_filtered = F.conv2d(pred, kernel, padding=self.padding)
            target_filtered = F.conv2d(target, kernel, padding=self.padding)
            loss += self.smooth_l1(pred_filtered, target_filtered)
        return loss / len(self.kernels)


class DistributionLoss(nn.Module):
    """Ошибка распределения"""
    def __init__(self, kernel_size=7, stride=2):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, target):
        pred_patches = self.unfold(pred)
        target_patches = self.unfold(target)
        pred_var = torch.var(pred_patches, dim=1, unbiased=False)
        target_var = torch.var(target_patches, dim=1, unbiased=False)
        return self.l1_loss(pred_var, target_var)


class MoireNetLoss(nn.Module):
    """Полная функция потерь"""
    def __init__(self, w1=1.0, w2=0.8, w3=0.8):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.pixel_loss = nn.SmoothL1Loss()
        self.direction_loss = DirectionLoss()
        self.distribution_loss = DistributionLoss()

    def forward(self, pred, target):
        L_pixel = self.pixel_loss(pred, target)
        L_dir = self.direction_loss(pred, target)
        L_dis = self.distribution_loss(pred, target)
        total = self.w1 * L_pixel + self.w2 * L_dir + self.w3 * L_dis

        return {
            'total': total,
            'pixel': L_pixel,
            'dir': L_dir,
            'dis': L_dis
        }


# =============================================================================
# 3. DENSE БЛОКИ (SADenseBlock, RSADB)
# =============================================================================
# SADenseBlock: Dense block с SimAM вниманием
# RSADB: Residual SADenseBlock
# =============================================================================

class SA_make_dense(nn.Module):
    """Conv + SimAM + Concat"""
    def __init__(self, nChannels, growthRate, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size,
                      padding=(kernel_size-1)//2 + dilation-1, stride=1, dilation=dilation),
            nn.SELU()
        )
        self.attn = simam_module()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.attn(outputs)
        return torch.cat((inputs, outputs), dim=1)


class SADenseBlock(nn.Module):
    """Dense block с SimAM"""
    def __init__(self, in_size, nDenselayer, growthRate):
        super().__init__()
        nChannels_ = in_size
        modules = []
        for _ in range(nDenselayer):
            modules.append(SA_make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nChannels_, in_size, kernel_size=1),
            nn.SELU()
        )

    def forward(self, inputs):
        outputs = self.dense_layers(inputs)
        return self.bottleneck(outputs)


class RSADB(nn.Module):
    """Residual dense block"""
    def __init__(self, in_size, nDenselayer, growthRate, dilation=1):
        super().__init__()
        nChannels_ = in_size
        modules = []
        for _ in range(nDenselayer):
            modules.append(SA_make_dense(nChannels_, growthRate, dilation=dilation))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, in_size, kernel_size=1)

    def forward(self, inputs):
        outputs = self.dense_layers(inputs)
        outputs = self.conv_1x1(outputs)
        return inputs + outputs


class PS_Upsample(nn.Module):
    """PixelShuffle апсемплинг"""
    def __init__(self, in_size):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_size, in_size * 2, 3, padding=1),
            nn.SELU(),
            nn.PixelShuffle(2)
        )
    def forward(self, inputs):
        return self.up(inputs)


# =============================================================================
# 4. SOBEL ГРАДИЕНТЫ
# =============================================================================

class Sobel_Grads(nn.Module):
    def __init__(self):
        super().__init__()
        gx = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype='float32')
        gy = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype='float32')
        self.register_buffer('gx', torch.from_numpy(gx).expand(3, 1, 3, 3).contiguous())
        self.register_buffer('gy', torch.from_numpy(gy).expand(3, 1, 3, 3).contiguous())

    def forward(self, inputs):
        Gx = F.conv2d(inputs, self.gx, padding=1, groups=3)
        Gy = F.conv2d(inputs, self.gy, padding=1, groups=3)
        return torch.sqrt(Gx * Gx + Gy * Gy + 1e-6)


# =============================================================================
# 5. PIXEL DOMAIN - CNN ветка
# =============================================================================
# CNN -> многомасштабная обработка -> пространственные признаки
# =============================================================================

class Pyramidnet(nn.Module):
    def __init__(self, out_features=64):
        super().__init__()
        filters = [32, 64, 128, 256, 512]
        self.grads = Sobel_Grads()

        # Downsample ветки
        self.p_128 = BasicConv(6, filters[0], 3, padding=1)
        self.p_64 = BasicConv(filters[0], filters[1], 2, stride=2, padding=0)
        self.p_32 = BasicConv(filters[1], filters[2], 2, stride=2, padding=0)
        self.p_16 = BasicConv(filters[2], filters[3], 2, stride=2, padding=0)
        self.p_8  = BasicConv(filters[3], filters[4], 2, stride=2, padding=0)

        # 1×1 свёртки для выравнивания каналов после skip-connections
        self.p_1x1_128 = nn.Sequential(nn.Conv2d(filters[0]*2, filters[0], 1), nn.SELU())
        self.p_1x1_64  = nn.Sequential(nn.Conv2d(filters[1]*2, filters[1], 1), nn.SELU())
        self.p_1x1_32  = nn.Sequential(nn.Conv2d(filters[2]*2, filters[2], 1), nn.SELU())
        self.p_1x1_16  = nn.Sequential(nn.Conv2d(filters[3]*2, filters[3], 1), nn.SELU())

        # Dense блоки на каждом уровне
        self.p_DB_128 = nn.Sequential(SADenseBlock(filters[0], 5, 16), RSADB(filters[0], 10, 32))
        self.p_DB_64  = nn.Sequential(SADenseBlock(filters[1], 5, 16), RSADB(filters[1], 10, 32))
        self.p_DB_32  = nn.Sequential(SADenseBlock(filters[2], 5, 16), RSADB(filters[2], 10, 32))
        self.p_DB_16  = nn.Sequential(SADenseBlock(filters[3], 5, 16), RSADB(filters[3], 5, 32))
        self.p_DB_8   = nn.Sequential(SADenseBlock(filters[4], 5, 16), RSADB(filters[4], 5, 32))

        # Апсемплинг
        self.p_up_8  = PS_Upsample(filters[4])
        self.p_up_16 = PS_Upsample(filters[3])
        self.p_up_32 = PS_Upsample(filters[2])
        self.p_up_64 = PS_Upsample(filters[1])

        # Финальная свёртка
        self.final_conv = nn.Conv2d(filters[0], out_features, 3, padding=1)

    def forward(self, inputs):
        # Sobel градиенты и конкатенация с исходным изображением
        p_grad = self.grads(inputs)
        p_inputs = torch.cat((p_grad, inputs), dim=1)

        # Downsample этап
        p128 = self.p_128(p_inputs)
        p64  = self.p_64(p128)
        p32  = self.p_32(p64)
        p16  = self.p_16(p32)
        p8   = self.p_8(p16)

        # Upsample этап со skip-connections
        db8  = self.p_DB_8(p8)
        up8  = self.p_up_8(db8)

        db16 = self.p_1x1_16(torch.cat([p16, up8], dim=1))
        db16 = self.p_DB_16(db16)
        up16 = self.p_up_16(db16)

        db32 = self.p_1x1_32(torch.cat([p32, up16], dim=1))
        db32 = self.p_DB_32(db32)
        up32 = self.p_up_32(db32)

        db64 = self.p_1x1_64(torch.cat([p64, up32], dim=1))
        db64 = self.p_DB_64(db64)
        up64 = self.p_up_64(db64)

        db128 = self.p_1x1_128(torch.cat([p128, up64], dim=1))
        db128 = self.p_DB_128(db128)

        return self.final_conv(db128)


# =============================================================================
# 6. FREQUENCY DOMAIN - частотная ветка
# =============================================================================
# DCT -> частотная обработка -> IDCT -> пространственные признаки
# =============================================================================

class Transform2DCT(nn.Module):
    """2D DCT"""
    def forward(self, x):
        return dct.dct_2d(x, norm='ortho')

class InverseDCT(nn.Module):
    """Обратное DCT"""
    def forward(self, x):
        return dct.idct_2d(x, norm='ortho')


class FrequencyNetwork(nn.Module):
    def __init__(self, out_features=64):
        super().__init__()
        
        self.Trans2DCT = Transform2DCT()
        self.IDCT = InverseDCT()
        
        self.dct_features = BasicConv(3, 32, 3, padding=1)
        
        self.SADB1 = SADenseBlock(32, 5, 16)
        self.RSADB1 = RSADB(32, 8, 32)
        self.RSADB2 = RSADB(32, 8, 32)
        self.RSADB3 = RSADB(32, 6, 32)
        
        # Выход — изображение (3 канала)
        self.to_image = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.SELU()
        )
        
        # Расширение каналов
        self.expand = nn.Conv2d(3, out_features, 1)

    def forward(self, x):
        # DCT в частотную область
        x_dct = self.Trans2DCT(x)
        
        # Нормализация
        x_dct = (x_dct - x_dct.mean(dim=[2,3], keepdim=True)) / \
                (x_dct.std(dim=[2,3], keepdim=True) + 1e-5)
        
        # Обработка в частотной области
        feat = self.dct_features(x_dct)
        feat = self.SADB1(feat)
        feat = self.RSADB1(feat)
        feat = self.RSADB2(feat)
        feat = self.RSADB3(feat)
        
        # Обратно в изображение
        img = self.to_image(feat)
        
        # Возврат в пространственную область
        img_spatial = self.IDCT(img)
        
        # Расширение признаков
        result = self.expand(img_spatial)
        
        return result


# =============================================================================
# 7. WAVELET DOMAIN - вейвлет ветка
# =============================================================================
# SWT -> wavelet обработка -> пространственно-частотные признаки
# =============================================================================

class WaveletBranch(nn.Module):
    def __init__(self, wavelet='haar', mode='symmetric', out_features=64):
        super().__init__()
        # SWT преобразование
        self.swt = SWTForward(J=1, wave=wavelet, mode=mode)
        
        # RGB -> grayscale
        self.register_buffer('rgb_weights', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        
        # Расширение каналов
        self.expand = nn.Conv2d(1, out_features, 1)
    
    def forward(self, x):
        with torch.no_grad():
            # Grayscale
            x_gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)
            
            # SWT
            coeffs = self.swt(x_gray)
            
            # Извлекаем LL, LH, HL
            ll = coeffs[0][:, 0:1, :, :]
            lh = coeffs[0][:, 1:2, :, :]
            hl = coeffs[0][:, 2:3, :, :]
            
            # Нормализация LL
            ll_min = ll.amin(dim=(2,3), keepdim=True)
            ll_max = ll.amax(dim=(2,3), keepdim=True)
            ll_norm = (ll - ll_min) / (ll_max - ll_min + 1e-8)
            
            # Нормализация LH, HL
            lh_max = lh.abs().amax(dim=(2,3), keepdim=True)
            hl_max = hl.abs().amax(dim=(2,3), keepdim=True)
            lh_norm = lh / (lh_max + 1e-8)
            hl_norm = hl / (hl_max + 1e-8)
            
            # Перемножение
            high = torch.max(lh_norm, hl_norm)
            result = ll_norm * high
            
            # Расширение признаков
            result = self.expand(result)
            
            return result

# =============================================================================
# 8. TRIPLE DYNAMIC FUSION - слияние трёх доменов
# =============================================================================

class TripleDynamicFusion(nn.Module):
    def __init__(self, in_features=64):
        super().__init__()

        # Объединение признаков
        self.reduce = nn.Sequential(
            nn.Conv2d(in_features * 3, in_features, 1),
            nn.SELU()
        )

        # SimAM attention
        self.attention = simam_module()

        # Depthwise conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=5,
                padding=2,
                groups=in_features
            ),
            nn.SELU()
        )

        # Pointwise conv
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_features, in_features, 1),
            nn.SELU()
        )

        # Предсказание маски
        self.to_mask = nn.Sequential(
            nn.Conv2d(in_features, 32, 3, padding=1),
            nn.SELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, p_feat, f_feat, w_feat):
        # Объединение доменов
        combined = torch.cat([p_feat, f_feat, w_feat], dim=1)
        
        # Сжатие каналов
        x = self.reduce(combined)

        # Attention
        x = self.attention(x)

        residual = x

        # Локальная обработка
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Residual connection
        x = x + residual

        # Финальная маска
        return self.to_mask(x)


# =============================================================================
# 9. ПОЛНАЯ МОДЕЛЬ
# =============================================================================

class MoireNetTripleDomain(nn.Module):
    """Трёхдоменная сеть для локализации муара"""
    def __init__(self, hidden_features=64):
        super().__init__()
        self.pixel_branch = Pyramidnet(out_features=hidden_features)
        self.freq_branch = FrequencyNetwork(out_features=hidden_features)
        self.wavelet_branch = WaveletBranch(out_features=hidden_features)
        self.fusion = TripleDynamicFusion(in_features=hidden_features)

    def forward(self, x):
        feat_p = self.pixel_branch(x)
        feat_f = self.freq_branch(x)
        feat_w = self.wavelet_branch(x)
        return self.fusion(feat_p, feat_f, feat_w)

# =============================================================================
# 10. ДАТАСЕТ И ОБУЧЕНИЕ
# =============================================================================

class MoireDataset(Dataset):
    """Загрузчик пар изображение-маска"""
    def __init__(self, root_dir, split='train', image_size=(320, 320),
                 use_augmentation=True, max_samples=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation and split == 'train'
        
        total = 20129 if split == 'train' else 1499
        self.indices = [f"{i:06d}" for i in range(1, total + 1)]
        
        if max_samples:
            self.indices = self.indices[:max_samples]
        
        print(f"Загружено {len(self.indices)} образцов из {split}")
        self.transform = self._get_transforms()

    # Предобработка
    def _get_transforms(self):
        if self.use_augmentation:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        folder_num = self.indices[idx]
        
        img_path = self.root_dir / self.split / str(folder_num) / "input.jpg"
        mask_path = self.root_dir / self.split / str(folder_num) / "mask.png"
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask'].unsqueeze(0).float()


def compute_l2(pred, target):
    """L2-метрика"""
    return torch.nn.functional.mse_loss(pred, target).sqrt().item()
    
def compute_correlation(pred, target):
    """Корреляция предсказания и target"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    pred_norm = pred_flat - pred_flat.mean()
    target_norm = target_flat - target_flat.mean()
    
    correlation = (pred_norm * target_norm).sum() / (pred_norm.norm() * target_norm.norm() + 1e-6)
    return correlation.item()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    total_pixel = 0
    total_dir = 0
    total_dis = 0

    for images, masks in tqdm(loader, desc="Обучение", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        pred = model(images)
        losses = criterion(pred, masks)
        loss = losses['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_pixel += losses['pixel'].item()
        total_dir += losses['dir'].item()
        total_dis += losses['dis'].item()

    n = len(loader)

    return {
        'loss': total_loss / n,
        'pixel': total_pixel / n,
        'dir': total_dir / n,
        'dis': total_dis / n
    }
def save_visualization(model, loader, device, name="epoch", vis_dir=None, num_samples=4):
    """Визуализация предсказаний модели"""
    model.eval()
    images, masks = next(iter(loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].cpu()
    with torch.no_grad():
        preds = model(images).cpu().detach()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
    for i in range(num_samples):
        img = images[i].cpu().permute(1,2,0).numpy()
        axes[i,0].imshow(img)
        axes[i,0].set_title("Input")
        axes[i,0].axis('off')
        axes[i,1].imshow(masks[i,0], cmap='gray')
        axes[i,1].set_title("GT")
        axes[i,1].axis('off')
        axes[i,2].imshow(preds[i,0], cmap='gray')
        axes[i,2].set_title("Pred")
        axes[i,2].axis('off')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/epoch_{name}.png", dpi=160, bbox_inches='tight')
    plt.close()

def validate(model, loader, criterion, device):
    """Валидация модели"""
    model.eval()

    total_loss = 0
    total_pixel = 0
    total_dir = 0
    total_dis = 0
    total_l2 = 0
    total_target_mean = 0
    total_correlation = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Валидация", leave=False):

            images, masks = images.to(device), masks.to(device)

            pred = model(images)

            losses = criterion(pred, masks)

            total_loss += losses['total'].item()
            total_pixel += losses['pixel'].item()
            total_dir += losses['dir'].item()
            total_dis += losses['dis'].item()
            total_target_mean += masks.mean().item()
            total_correlation += compute_correlation(pred, masks)
            

            total_l2 += compute_l2(pred, masks)

    n = len(loader)

    return {
        'loss': total_loss / n,
        'pixel': total_pixel / n,
        'dir': total_dir / n,
        'dis': total_dis / n,
        'l2': total_l2 / n,
        'target_mean': total_target_mean / n,
        'correlation': total_correlation / n
    }


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    # Пути
    DATA_ROOT = "/path/to/pairs/"
    SAVE_DIR = "./results/checkpoints"
    VIS_DIR = "./visualise"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Датасет
    train_dataset = MoireDataset(DATA_ROOT, split='train', image_size=(320, 320), use_augmentation=False)
    val_dataset = MoireDataset(DATA_ROOT, split='val', image_size=(320, 320), use_augmentation=False)

    # Загрузчик
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Модель
    model = MoireNetTripleDomain(hidden_features=64).to(device)

    # Loss
    criterion = MoireNetLoss(w1=1.0, w2=0.8, w3=0.8)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_l2 = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_l2': []}
    
    # Обучение
    for epoch in range(1, 51):
        print(f"\nЭпоха {epoch}/50")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Сохранение визуализации каждые 5 эпох
        if epoch % 5 == 0:
            save_visualization(model, val_loader, device, epoch, VIS_DIR)
        scheduler.step()

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_l2'].append(val_metrics['l2'])

        # Метрики на тренировочной выборке
        print(
        f"Train: "
        f"total={train_metrics['loss']:.5f} | "
        f"pixel={train_metrics['pixel']:.5f} | "
        f"dir={train_metrics['dir']:.5f} | "
        f"dis={train_metrics['dis']:.5f}"
        )

        # Метрики на валидационной выборке
        print(
        f"Val: "
        f"total={val_metrics['loss']:.5f} | "
        f"pixel={val_metrics['pixel']:.5f} | "
        f"dir={val_metrics['dir']:.5f} | "
        f"dis={val_metrics['dis']:.5f} | "
        f"L2={val_metrics['l2']:.5f}"
        )

        # Сохранение лучшей модели
        if val_metrics['l2'] < best_l2:
            best_l2 = val_metrics['l2']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"→ Лучшая модель сохранена (L2 = {best_l2:.4f})")

    # Сохранение финальной модели
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pth'))
    print(f"\nОбучение завершено! Лучший L2 = {best_l2:.4f}")

    # Визуализация лучшей модели
    best_path = os.path.join(SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        save_visualization(model, val_loader, device, name="best", vis_dir=VIS_DIR)
        print(f"best.png сохранён")
    else:
        print("best_model.pth не найден")

    # Визуализация финальной модели
    final_path = os.path.join(SAVE_DIR, 'final_model.pth')
    model.load_state_dict(torch.load(final_path, map_location=device))
    save_visualization(model, val_loader, device, name="final", vis_dir=VIS_DIR)
    print(f"final.png сохранён")

    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].legend()
    axes[0].set_title('Функция потерь')

    axes[1].plot(history['val_l2'], label='L2', color='red')
    axes[1].legend()
    axes[1].set_title('Метрика L2')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
    plt.close()


if __name__ == "__main__":
    main()