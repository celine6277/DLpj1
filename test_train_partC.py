"""
Training script for MNIST classification with MLP and CNN models.
Supports learning curve logging and confusion matrix generation for error analysis.
"""

import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# =====================================================
# HYPERPARAMETER CONFIGURATION (Part C: Optimization)
# =====================================================
LEARNING_RATE = 0.06  # Expose this for easy modification
BATCH_SIZE = 32
NUM_EPOCHS = 10
LOG_ITERS = 500
SAVE_DIR = './best_models'

# Model selection: 'MLP' or 'CNN'
MODEL_TYPE = 'MLP'  

# Fixed seed for reproducibility
np.random.seed(309)

# =====================================================
# DATA LOADING
# =====================================================
print("Loading MNIST dataset...")

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'
test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

def load_mnist_data(images_path, labels_path):
    """Load MNIST data from gzip files."""
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

# Load training and test data
train_imgs, train_labs = load_mnist_data(train_images_path, train_labels_path)
test_imgs, test_labs = load_mnist_data(test_images_path, test_labels_path)

print(f"Train data shape: {train_imgs.shape}, labels shape: {train_labs.shape}")
print(f"Test data shape: {test_imgs.shape}, labels shape: {test_labs.shape}")

# Split training data into train and validation sets
num_train = train_imgs.shape[0]
idx = np.random.permutation(np.arange(num_train))

# Save the index for reproducibility
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, 'idx.pickle'), 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# Normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0
test_imgs = test_imgs / 255.0

print(f"Normalized train shape: {train_imgs.shape}")
print(f"Normalized validation shape: {valid_imgs.shape}")

# =====================================================
# MODEL INSTANTIATION
# =====================================================
print(f"\nBuilding {MODEL_TYPE} model...")

if MODEL_TYPE == 'MLP':
    # MLP: 784 -> 600 -> 10
    model = nn.models.Model_MLP(
        size_list=[train_imgs.shape[-1], 600, 10], 
        act_func='ReLU'
        # lambda_list=[1e-4, 1e-4]
    )
elif MODEL_TYPE == 'CNN':
    # CNN model
    model = nn.models.Model_CNN()
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# =====================================================
# OPTIMIZER AND LOSS SETUP
# =====================================================
optimizer = nn.optimizer.MomentGD(init_lr=LEARNING_RATE, model=model, mu=0.9)
scheduler = nn.lr_scheduler.MultiStepLR(
    optimizer=optimizer, 
    milestones=[800, 2400, 4000], 
    gamma=0.5
)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)

# =====================================================
# DATA LOGGING FOR LEARNING CURVES (Part C: Visualization)
# =====================================================
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': []
}

# =====================================================
# TRAINING LOOP
# =====================================================
print(f"\nStarting training with {MODEL_TYPE} model...")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Epochs: {NUM_EPOCHS}\n")

runner = nn.runner.RunnerM(
    model, 
    optimizer, 
    nn.metric.accuracy, 
    loss_fn, 
    batch_size=BATCH_SIZE,
    scheduler=scheduler
)

runner.train(
    [train_imgs, train_labs], 
    [valid_imgs, valid_labs], 
    num_epochs=NUM_EPOCHS, 
    log_iters=LOG_ITERS, 
    save_dir=SAVE_DIR,
    save_name='MLP_Momentum_best_model.pickle' 
)


# =====================================================
# DATA LOGGING: Extract epoch-level metrics
# =====================================================
# The runner stores per-iteration metrics; we'll aggregate to per-epoch
batch_per_epoch = int(np.ceil(train_imgs.shape[0] / BATCH_SIZE))
dev_logs_per_epoch = int(np.ceil(batch_per_epoch / LOG_ITERS))
for epoch in range(NUM_EPOCHS):
    # Train 
    start_idx = epoch * batch_per_epoch
    end_idx = (epoch + 1) * batch_per_epoch
    epoch_train_loss = np.mean(runner.train_loss[start_idx:end_idx])
    epoch_train_acc = np.mean(runner.train_scores[start_idx:end_idx])
    
    # Validation 的切片逻辑单独计算
    start_dev_idx = epoch * dev_logs_per_epoch
    end_dev_idx = (epoch + 1) * dev_logs_per_epoch
    epoch_val_loss = np.mean(runner.dev_loss[start_dev_idx:end_dev_idx])
    epoch_val_acc = np.mean(runner.dev_scores[start_dev_idx:end_dev_idx])

    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    history['epochs'].append(epoch + 1)
    
    print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, "
          f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")

# =====================================================
# TESTING AND ERROR ANALYSIS (Part C)
# =====================================================
print("\n" + "="*60)
print("TESTING AND ERROR ANALYSIS")
print("="*60 + "\n")

print("Evaluating on test set...")
test_logits = model(test_imgs)
test_preds = np.argmax(test_logits, axis=1)
test_acc = np.mean(test_preds == test_labs)

print(f"Test Accuracy: {test_acc:.4f}")

# Compute confusion matrix
print("\nGenerating confusion matrix for test set...")
cm = confusion_matrix(test_labs, test_preds)
print(f"Confusion Matrix shape: {cm.shape}")
print(cm)

# Save confusion matrix as image
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title(f'{MODEL_TYPE} Model - Confusion Matrix on Test Set')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f'{MODEL_TYPE}_confusion_matrix.png'), dpi=100)
print(f"Confusion matrix saved to {os.path.join(SAVE_DIR, f'{MODEL_TYPE}_confusion_matrix.png')}")

# =====================================================
# LEARNING CURVE VISUALIZATION
# =====================================================
print("\nGenerating learning curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(history['epochs'], history['train_loss'], 'b-o', label='Train Loss')
ax1.plot(history['epochs'], history['val_loss'], 'r-s', label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title(f'{MODEL_TYPE} Model - Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy curve
ax2.plot(history['epochs'], history['train_acc'], 'b-o', label='Train Accuracy')
ax2.plot(history['epochs'], history['val_acc'], 'r-s', label='Val Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title(f'{MODEL_TYPE} Model - Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, f'{MODEL_TYPE}_learning_curves.png'), dpi=100)
print(f"Learning curves saved to {os.path.join(SAVE_DIR, f'{MODEL_TYPE}_learning_curves.png')}")

# Save history for later use
with open(os.path.join(SAVE_DIR, 'MLP_Momentum_history.pickle'), 'wb') as f:
    pickle.dump(history, f)
print(f"Training history saved to {os.path.join(SAVE_DIR, 'MLP_Momentum_history.pickle')}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)