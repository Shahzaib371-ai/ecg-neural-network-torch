"""
Task 4 - Train (Custom Mini-Batch Gradient Descent)
Run from VS Code: python /home/shahzaibhasnain/Work/task4.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- Imports from previous tasks ---
# task1.py se exact wahi variables import kar rahe hain jo wahan define hue hain
from task1 import X_train, y_train, X_val, y_val
from task2 import ECGNet, cross_entropy

def train(alpha, train_x, train_y, val_x, val_y):
    # Requirement: Network create karne se pehle seed 0 set karna hai
    torch.manual_seed(0)
    model = ECGNet()
    
    # Requirement: Batch size 128 aur shuffle=True
    batch_size = 128
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Hyperparameters
    beta = 0.9      # Momentum
    lambda_wd = 1e-4 # Weight decay
    epochs = 30     # Epochs
    
    # Har parameter ke liye velocity ko zero se initialize karein
    velocities = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    
    # Metrics record karne ke liye lists
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            # 1. Forward pass
            logits = model(batch_x)
            
            # 2. Compute Loss
            loss = cross_entropy(logits, batch_y)
            
            # 3. Backward pass
            loss.backward()
            
            # 4. Custom Update Rule
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                        
                    g = p.grad
                    
                    # Weight decay (L2) sirf weights par apply hoga (biases par nahi)
                    if p.ndim > 1:  
                        g = g + lambda_wd * p
                        
                    # Velocity update: v = beta*v + (1-beta)*g
                    velocities[name] = beta * velocities[name] + (1 - beta) * g
                    
                    # Parameter update: P = P - alpha * v
                    p -= alpha * velocities[name]
                    
                    # Gradients zero karna zaroori hai agle batch ke liye
                    p.grad.zero_()
                    
            # Loss aur accuracy track karein
            epoch_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        # --- Validation Phase (Har epoch ke baad) ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = cross_entropy(val_logits, val_y).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == val_y).sum().item() / val_y.size(0)
            
        train_loss = epoch_loss / total_train
        train_acc = correct_train / total_train
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
    return model, history

if __name__ == "__main__":
    # Ensure results directory exists
    RESULTS_DIR = "/home/shahzaibhasnain/Work/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Learning rates jo test karni hain
    alphas = [0.01, 0.1, 0.5]
    all_histories = {}
    models = {}

    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        print(f"Training with alpha = {alpha}...")
        # X_train, y_train, X_val, y_val ab task1.py se dynamically pass ho rahe hain
        trained_model, hist = train(alpha, X_train, y_train, X_val, y_val)
        all_histories[alpha] = hist
        models[alpha] = trained_model
        
        # Task 4.2: Plot validation loss against epoch for all three
        plt.plot(range(1, 31), hist['val_loss'], label=f'alpha = {alpha}')

    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss for different Learning Rates')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/val_loss_comparison.png')
    print("Graph saved: val_loss_comparison.png")
    plt.close()

    # --- Best Alpha Selection aur Graphing (Task 4.3) ---
    best_alpha = min(alphas, key=lambda a: min(all_histories[a]['val_loss']))
    print(f"\nBest alpha chosen: {best_alpha}")

    best_hist = all_histories[best_alpha]
    best_model = models[best_alpha]

    # Plot Training vs Validation Loss for chosen alpha
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), best_hist['train_loss'], label='Train Loss')
    plt.plot(range(1, 31), best_hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for best alpha ({best_alpha})')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/best_alpha_loss.png')
    plt.close()

    # Plot Training vs Validation Accuracy for chosen alpha
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), best_hist['train_acc'], label='Train Accuracy')
    plt.plot(range(1, 31), best_hist['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curves for best alpha ({best_alpha})')
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/best_alpha_acc.png')
    plt.close()

    # Task 4.3: Best model ke parameters save karna
    model_path = f"{RESULTS_DIR}/best_ecgnet.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model parameters saved to {model_path} successfully!")