"""
Task 5 - Test once
Run from VS Code: python /home/shahzaibhasnain/Work/task5.py
"""
import torch
import numpy as np

# task1.py se test data aur class names import karein
from task1 import X_test_t, y_test_t, CLASS_NAMES

# task2.py se model import karein
from task2 import ECGNet

# --- Task 5.2: Metric Functions ---
def accuracy(preds, y):
    """Overall test accuracy"""
    correct = (preds == y).sum().item()
    total = y.size(0)
    return correct / total

def confusion_matrix(preds, y, num_classes=5):
    """Confusion matrix where rows = true classes, columns = predicted classes"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    # Loop chalaney ke liye tensors ko 1D numpy arrays mein convert kar rahe hain
    y_np = y.numpy()
    preds_np = preds.numpy()
    
    for i in range(len(y_np)):
        true_class = y_np[i]
        pred_class = preds_np[i]
        cm[true_class, pred_class] += 1
        
    return cm

def per_class_accuracy(cm):
    """Calculate accuracy for each class individually"""
    num_classes = cm.shape[0]
    accuracies = []
    for i in range(num_classes):
        total_true_for_class = cm[i, :].sum() # Class i ki total actual values (Row sum)
        correct_for_class = cm[i, i]          # Class i ki correct predictions (Diagonal element)
        if total_true_for_class > 0:
            acc = correct_for_class / total_true_for_class
        else:
            acc = 0.0
        accuracies.append(acc)
    return accuracies

def main():
    # --- Task 5.1: Load saved parameters and predict ---
    print("Loading the best model...")
    model = ECGNet()
    
    # Task 4 se jo model save hua tha uska path
    model_path = "/home/shahzaibhasnain/Work/results/best_ecgnet.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval() # Model ko evaluation mode mein set karein
    
    print("Running predictions on test set...\n")
    with torch.no_grad():
        logits = model(X_test_t)
        # Class with the largest logit is the prediction
        preds = torch.argmax(logits, dim=1)
        
    # --- Task 5.3: Report Metrics ---
    # 1. Overall Test Accuracy
    test_acc = accuracy(preds, y_test_t)
    print(f"=== Test Accuracy ===")
    print(f"Overall Accuracy: {test_acc * 100:.2f}%")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(preds, y_test_t)
    print("\n=== Confusion Matrix ===")
    print("(Rows: True Classes, Columns: Predicted Classes)")
    print(f"{'':>4} | {'0':>5} {'1':>5} {'2':>5} {'3':>5} {'4':>5}")
    print("-" * 38)
    for i in range(5):
        row_str = " ".join([f"{val:5d}" for val in cm[i]])
        print(f"{i:>4} | {row_str}")
        
    # 3. Per-Class Accuracy
    per_class_accs = per_class_accuracy(cm)
    print("\n=== Per-Class Accuracy ===")
    for i in range(5):
        # Class name truncate kar rahe hain taake display neat lagay
        print(f"Class {i} ({CLASS_NAMES[i][:15]:<15}): {per_class_accs[i] * 100:.2f}%")
        
    # 4. Balanced Accuracy
    balanced_acc = sum(per_class_accs) / len(per_class_accs)
    print(f"\n=== Balanced Accuracy ===")
    print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")

if __name__ == "__main__":
    main()