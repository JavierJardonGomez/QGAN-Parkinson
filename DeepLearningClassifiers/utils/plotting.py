import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score, classification_report, confusion_matrix

def plot_training(history, output_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Train Acc')
    plt.plot(history["val_acc"], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(f"{output_name}_training.png")
    #plt.show()

def plot_classification_report(y_true, y_pred, labels, output_name=None):
    # 1) Obtener el informe como dict para convertirlo en DataFrame
    report_dict = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    # 2) Imprimir texto en consola (igual que antes)
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    # 3) Dibujar la tabla en Matplotlib
    fig, ax = plt.subplots(figsize=(8, len(report_df)*0.5 + 1))
    ax.axis('off')
    tbl = ax.table(
        cellText=np.round(report_df.values, 2),
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.title("Classification Report", pad=20)
    plt.tight_layout()
    
    # 4) Guardar si nos pasan un nombre de salida
    if output_name is not None:
        fig.savefig(f"{output_name}_classification_report.png")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, labels, output_name):
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    plt.title("Matriz de Confusión")
    plt.savefig(f"{output_name}_confusion_matrix.png")
    plt.show()
    plt.close()  

def plot_roc_curve(y_true, y_probs, output_name):
    plt.figure()
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_name}_roc_curve.png")
    plt.show()
    plt.close()