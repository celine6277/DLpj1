"""
Robustness Analysis for MNIST Models.
Evaluates model stability under Gaussian Noise, Translation, and Rotation.
Generates degradation curves for engineering-level comparison.
"""

import mynn as nn
import numpy as np
import gzip
from struct import unpack
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate
import pickle

# =====================================================
# CONFIGURATION
# =====================================================
SAVE_DIR = './best_models'
# 如果 CNN 还没跑完，这里可以先只留 'MLP' 测试代码
MODELS_TO_TEST = ['MLP', 'CNN'] # 

# =====================================================
# PERTURBATION FUNCTIONS (Engineering Standard)
# =====================================================

def add_gaussian_noise(imgs, std):
    """
    向量化添加高斯噪声
    imgs: [N, 784]
    """
    noise = np.random.normal(loc=0.0, scale=std, size=imgs.shape)
    noisy_imgs = imgs + noise
    return np.clip(noisy_imgs, 0., 1.)

def translate_images(imgs, shift_pixels):
    """
    纯 NumPy 向量化平移 (Padding + Cropping)
    shift_pixels: 正数向右下平移，负数向左上平移
    """
    if shift_pixels == 0:
        return imgs.copy()
        
    imgs_2d = imgs.reshape(-1, 28, 28)
    shift = abs(shift_pixels)
    
    # 填充四周
    pad_width = ((0, 0), (shift, shift), (shift, shift))
    padded = np.pad(imgs_2d, pad_width, mode='constant', constant_values=0)
    
    # 根据平移方向裁剪
    if shift_pixels > 0: # 向右下
        shifted = padded[:, 0:28, 0:28]
    else:                # 向左上
        shifted = padded[:, 2*shift:2*shift+28, 2*shift:2*shift+28]
        
    return shifted.reshape(-1, 784)

def rotate_images(imgs, angle):
    """
    利用 scipy 批量旋转图片
    """
    if angle == 0:
        return imgs.copy()
        
    imgs_2d = imgs.reshape(-1, 28, 28)
    # axes=(1,2) 表示在空间维度 H和W 上旋转，保持 Batch 维度不变
    rotated = rotate(imgs_2d, angle, axes=(1, 2), reshape=False, mode='constant', cval=0.0)
    return np.clip(rotated, 0., 1.).reshape(-1, 784)


# =====================================================
# DATA LOADING & EVALUATION PIPELINE
# =====================================================
def evaluate_robustness():
    print("Loading test dataset...")
    test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
    test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    test_imgs = test_imgs / 255.0

    # 加载模型字典
    models = {}
    for model_name in MODELS_TO_TEST:
        try:
            print(f"Loading {model_name} model...")
            if model_name == 'MLP':
                # 这里加载你跑出来的带有 Momentum 的 best model，性能更好
                model = nn.models.Model_MLP(size_list=[784, 600, 10], act_func='ReLU')
                # 注意：如果你的模型保存逻辑改了，这里路径也要对应
                model.load_model(os.path.join(SAVE_DIR, 'MLP_Momentum_best_model.pickle')) 
            elif model_name == 'CNN':
                model = nn.models.Model_CNN()
                model.load_model(os.path.join(SAVE_DIR, 'CNN_best_model.pickle'))
            models[model_name] = model
        except FileNotFoundError:
            print(f"Warning: {model_name} model file not found. Skipping...")

    # 定义测试区间 (Perturbation Intensities)
    noise_stds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    translations = [0, 1, 2, 3, 4, 5]  # 偏移的像素点
    rotations = [0, 10, 20, 30, 40, 50] # 旋转的角度

    results = {
        'Noise': {m: [] for m in models.keys()},
        'Translation': {m: [] for m in models.keys()},
        'Rotation': {m: [] for m in models.keys()}
    }

    # 1. 评估 Gaussian Noise
    print("\n--- Evaluating Gaussian Noise ---")
    for std in noise_stds:
        noisy_imgs = add_gaussian_noise(test_imgs, std)
        for name, model in models.items():
            preds = np.argmax(model(noisy_imgs), axis=1)
            acc = np.mean(preds == test_labs)
            results['Noise'][name].append(acc)
        print(f"Noise Std {std}: " + ", ".join([f"{n}: {results['Noise'][n][-1]:.4f}" for n in models.keys()]))

    # 2. 评估 Translation
    print("\n--- Evaluating Translation ---")
    for shift in translations:
        shifted_imgs = translate_images(test_imgs, shift)
        for name, model in models.items():
            preds = np.argmax(model(shifted_imgs), axis=1)
            acc = np.mean(preds == test_labs)
            results['Translation'][name].append(acc)
        print(f"Shift {shift}px: " + ", ".join([f"{n}: {results['Translation'][n][-1]:.4f}" for n in models.keys()]))

    # 3. 评估 Rotation
    print("\n--- Evaluating Rotation ---")
    for angle in rotations:
        rotated_imgs = rotate_images(test_imgs, angle)
        for name, model in models.items():
            preds = np.argmax(model(rotated_imgs), axis=1)
            acc = np.mean(preds == test_labs)
            results['Rotation'][name].append(acc)
        print(f"Angle {angle}°: " + ", ".join([f"{n}: {results['Rotation'][n][-1]:.4f}" for n in models.keys()]))

    # =====================================================
    # PLOTTING
    # =====================================================
    print("\nGenerating Robustness Degradation Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Noise Plot
    for name in models.keys():
        axes[0].plot(noise_stds, results['Noise'][name], marker='o', label=name)
    axes[0].set_title('Robustness to Gaussian Noise')
    axes[0].set_xlabel('Noise Standard Deviation')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].grid(True)
    axes[0].legend()

    # Translation Plot
    for name in models.keys():
        axes[1].plot(translations, results['Translation'][name], marker='s', label=name)
    axes[1].set_title('Robustness to Translation')
    axes[1].set_xlabel('Shift (Pixels)')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].grid(True)
    axes[1].legend()

    # Rotation Plot
    for name in models.keys():
        axes[2].plot(rotations, results['Rotation'][name], marker='^', label=name)
    axes[2].set_title('Robustness to Rotation')
    axes[2].set_xlabel('Rotation Angle (Degrees)')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Robustness_Analysis.png'), dpi=150)
    print(f"Plot saved to {os.path.join(SAVE_DIR, 'Robustness_Analysis.png')}")

if __name__ == "__main__":
    evaluate_robustness()