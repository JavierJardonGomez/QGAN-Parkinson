import torch
import pandas as pd

from utils.plotting import plot_training
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def train_one_epoch(model, loader, criterion, optimizer, device, model_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        if model_name == "ffnn":
            inputs = inputs.view(inputs.size(0), -1).to(device)
        elif model_name == "lstm_fcn" or model_name == "conv":
            inputs = inputs.to(device)
        else:
            inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

    train_loss = running_loss / len(loader.dataset)
    train_acc = correct / total
    
    return train_loss, train_acc

def validate_train(model, test_loader, criterion, device, model_name):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if model_name == "ffnn":
                inputs = inputs.view(inputs.size(0), -1).to(device)
            elif model_name == "lstm_fcn" or model_name == "conv":
                inputs = inputs.to(device)
            else:
                inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

        test_loss = val_loss / len(test_loader.dataset)
        test_accuracy = correct / total

        print(f"Pérdida de test: {test_loss}")
        print(f"Precisión de test: {test_accuracy}")

        return test_loss, test_accuracy


def validate_real_data(model, real_test_loader, device, model_name):
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in real_test_loader:
            if model_name == "ffnn":
                    inputs = inputs.view(inputs.size(0), -1).to(device)
            elif model_name == "lstm_fcn" or model_name == "conv":
                inputs = inputs.to(device)
            else:
                inputs = inputs.permute(0, 2, 1).to(device)   
            labels = labels.to(device)  
            outputs = model(inputs)
            probs = torch.exp(outputs)[:, 1].detach().cpu().numpy()  
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()  
            true_labels = torch.argmax(labels, dim=1).detach().cpu().numpy()  

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(true_labels)

        return all_probs, all_preds, all_labels

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def validate_with_outputs(model, loader, criterion, device, model_name):
    model.eval()
    total_loss = 0.0

    all_true = []
    all_pred = []
    all_prob_pos = []  # prob de clase 1

    with torch.no_grad():
        for inputs, labels in loader:
            # inputs
            if model_name == "ffnn":
                inputs = inputs.view(inputs.size(0), -1).to(device)
            elif model_name in ["lstm_fcn", "conv"]:
                inputs = inputs.to(device)
            else:
                inputs = inputs.permute(0, 2, 1).to(device)

            labels = labels.to(device)  # one-hot
            y_true = labels.argmax(dim=1)  # 0/1

            outputs = model(inputs)  # logits (B,2)
            loss = criterion(outputs, y_true)
            total_loss += loss.item() * inputs.size(0)

            # preds
            y_pred = torch.argmax(outputs, dim=1)

            # probs (softmax) -> clase positiva = índice 1
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_true.append(y_true.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())
            all_prob_pos.append(probs.cpu().numpy())

    test_loss = total_loss / len(loader.dataset)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob_pos)

    test_acc = accuracy_score(y_true, y_pred)
    test_f1  = f1_score(y_true, y_pred)

    # AUC necesita las dos clases presentes
    test_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

    print(f"Test loss: {test_loss:.6f}")
    print(f"Test acc : {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

    return test_loss, test_acc, test_f1, test_auc, y_true, y_pred, y_prob

def bootstrap_ci_binary(y_true, y_pred, y_prob, n_boot=2000, alpha=0.05, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    acc_bs, f1_bs, auc_bs = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # con reemplazo
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_prob[idx]

        acc_bs.append(accuracy_score(yt, yp))
        f1_bs.append(f1_score(yt, yp))

        # AUC a veces no está definido si solo sale una clase
        if len(np.unique(yt)) == 2:
            auc_bs.append(roc_auc_score(yt, ys))

    def ci(samples):
        lo = np.quantile(samples, alpha/2)
        hi = np.quantile(samples, 1-alpha/2)
        return lo, hi

    return {
        "acc_ci": ci(acc_bs),
        "f1_ci":  ci(f1_bs),
        "auc_ci": ci(auc_bs) if len(auc_bs) > 0 else (np.nan, np.nan)
    }

from torch.utils.data import DataLoader, TensorDataset

def evaluate_metrics(model, inputs, labels, device, model_name, eval_bs: int = 16):
    model.eval()

    ds = TensorDataset(inputs, labels)
    dl = DataLoader(ds, batch_size=eval_bs, shuffle=False)

    all_preds = []
    all_true  = []

    with torch.no_grad():
        for xb, yb in dl:
            if model_name == "ffnn":
                xb = xb.view(xb.size(0), -1).to(device)
            elif model_name in ("lstm_fcn", "conv"):
                xb = xb.to(device)
            else:
                xb = xb.permute(0, 2, 1).to(device)

            yb = yb.to(device)

            out = model(xb)
            pred = torch.argmax(out, dim=1).detach().cpu().numpy()
            true = torch.argmax(yb, dim=1).detach().cpu().numpy()

            all_preds.append(pred)
            all_true.append(true)

    preds = np.concatenate(all_preds)
    true  = np.concatenate(all_true)

    precision = precision_score(true, preds, average="macro", zero_division=0)
    recall    = recall_score(true, preds, average="macro", zero_division=0)
    f1        = f1_score(true, preds, average="macro", zero_division=0)
    acc       = accuracy_score(true, preds)

    return acc, precision, recall, f1

"""
def evaluate_metrics(model, inputs, labels, device, model_name):
    model.eval()
    with torch.no_grad():
        
        if model_name == "ffnn":
            inputs = inputs.view(inputs.size(0), -1).to(device)
        elif model_name == "lstm_fcn" or model_name == "conv":
            inputs = inputs.to(device)
        else:
            inputs = inputs.permute(0, 2, 1).to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true = torch.argmax(labels, dim=1).cpu().numpy()

    precision = precision_score(true, preds, average="macro")
    recall = recall_score(true, preds, average="macro")
    f1 = f1_score(true, preds, average="macro")
    acc = accuracy_score(true, preds)

    return acc, precision, recall, f1    
""" 

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_name, 
               epochs=80, patience=10, save_path='best_model.pth', plot_path='plot.png'):
    best_val_loss = float('inf')
    no_improve = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, model_name)
        val_loss, val_acc = validate_train(model, val_loader, criterion, device, model_name)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path) #mejorar
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies
    }

    plot_training(history, plot_path)

    return model, history
