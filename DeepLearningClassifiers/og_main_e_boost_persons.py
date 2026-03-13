#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from data.processing_data_raw import read_audios_dir, load_data_raw
from models.factory import get_model
from train.training_utils import (
    train_loop,
    evaluate_metrics,
    validate_real_data,
    bootstrap_ci_binary,
)
from utils.plotting import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)
from utils.person_split_utils import (
    base_id,
    k_fold_split_by_person_seeded,
    person_id,
    select_people_with_3_variants,
    write_blind_log,
    write_cv_folds_log,
)

# --------------------------- Configuración global ----------------------------
vowel = "E"
#seconds_min = 0.49
seconds_min = 0.57
audio_dir_name = f"/home/jovyan/QVocoder/paper/data/resample_vowels/{vowel}"

experiment_ids = [40, 41, 42, 43]
batch_sizes = [8, 16, 32, 64]
lrs = [1e-5, 2e-5, 5e-5, 8e-5]
weight_decays = [1e-4, 1e-4, 1e-4, 1e-4]

TARGET_LEN = int(seconds_min * 22_050)
all_model_names = ["resnet", "inception", "conv", "lstm_fcn", "times_net"]

BASE_RESULTS_DIR = "og_results_e_1_boost_person"

# Blind-test selection (trazabilidad por persona)
SEED_BLIND = 42
N_PEOPLE_PER_CLASS = 10  # total personas blind = 20; total audios blind = 20*3 = 60

# CV determinista por persona
CV_K = 5
#CV_SEED = 42  # puedes usar SEED_BLIND si quieres (aquí lo igualamos por defecto)
CV_SEED = SEED_BLIND #same as blind


# --------------------------- Dispositivo -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Dispositivo detectado: {device}\n")

# -------------------------------- Main loop ----------------------------------
for exp_id, bs, lr, wd in zip(experiment_ids, batch_sizes, lrs, weight_decays):
    print(f"=== Experimento {exp_id} | BS={bs}, LR={lr}, WD={wd} ===")

    exp_dir = os.path.join(
        BASE_RESULTS_DIR,
        f"resample_vowels_{vowel}_{exp_id}_bs{bs}_lr{lr}_wd{wd}",
    )
    os.makedirs(exp_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Cargar listas de archivos
    # -------------------------------------------------------------------------
    audio_dir = audio_dir_name
    real_files, generated_files, label_dict = read_audios_dir(audio_dir)

    # -------------------------------------------------------------------------
    # 1.a Blind test: 10 personas control + 10 personas patológico => 60 audios
    # -------------------------------------------------------------------------
    real_files_clean = [f for f in real_files if "_generated" not in f]
    real_control = [f for f in real_files_clean if f.endswith("_c")]
    real_patologico = [f for f in real_files_clean if f.endswith("_p")]

    rng = random.Random(SEED_BLIND)

    chosen_ctrl_pids, ctrl_val_files = select_people_with_3_variants(
        real_control, N_PEOPLE_PER_CLASS, rng
    )
    chosen_pat_pids, pat_val_files = select_people_with_3_variants(
        real_patologico, N_PEOPLE_PER_CLASS, rng
    )

    real_val_names = ctrl_val_files + pat_val_files
    real_train_names = [f for f in real_files_clean if f not in set(real_val_names)]

    write_blind_log(
        exp_dir,
        SEED_BLIND,
        N_PEOPLE_PER_CLASS,
        chosen_ctrl_pids,
        chosen_pat_pids,
        ctrl_val_files,
        pat_val_files,
        real_val_names,
    )

    # -------------------------------------------------------------------------
    # 1.b Filtrar sintéticos por PERSONA (evita fugas con blind test)
    # -------------------------------------------------------------------------
    val_person_ids = {person_id(f) for f in real_val_names}

    filtered_generated_files = [
        f for f in generated_files
        if person_id(base_id(f)) not in val_person_ids
    ]

    train_names = real_train_names + filtered_generated_files

    # -------------------------------------------------------------------------
    # 2. Vectores de grupos (para CV): AGRUPAR POR PERSONA (a1/a2/a3 juntos)
    # -------------------------------------------------------------------------
    groups = np.array([person_id(base_id(f)) for f in train_names])  # (N,)

    # Log de folds CV deterministas (trazabilidad)
    write_cv_folds_log(groups, exp_dir, k=CV_K, seed=CV_SEED)

    # -------------------------------------------------------------------------
    # 3. Carga de características y etiquetas (alineadas)
    # -------------------------------------------------------------------------
    X_train, y_train_labels = load_data_raw(
        train_names, label_dict, audio_dir, TARGET_LEN
    )
    X_test, y_test_labels = load_data_raw(
        real_val_names, label_dict, audio_dir, TARGET_LEN
    )

    # One-hot global
    enc = OneHotEncoder(sparse_output=False)
    y_onehot = enc.fit_transform(y_train_labels.reshape(-1, 1))  # (N,2)

    # Tensores
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(
        enc.transform(y_test_labels.reshape(-1, 1)), dtype=torch.float32
    )

    # Comprobación opcional
    for i in range(min(5, len(train_names))):
        print(train_names[i], y_train_labels[i], "pid=", groups[i])

    # -------------------------------------------------------------------------
    # 4. CV por modelo (determinista por persona)
    # -------------------------------------------------------------------------
    for model_name in all_model_names:
        print(f"\n--- Modelo: {model_name} ---")
        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        output_name = os.path.join(model_dir, f"resample_vowels_{vowel}_{exp_id}_{model_name}")

        fold_metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
        best_val_acc = 0.0
        best_model_path = ""

        for fold, (tr_idx, val_idx) in enumerate(
            k_fold_split_by_person_seeded(groups, k=CV_K, seed=CV_SEED), start=1
        ):
            print(f"Fold {fold}: {len(tr_idx)} train - {len(val_idx)} val")

            X_tr, X_val = X_tensor[tr_idx], X_tensor[val_idx]
            y_tr, y_val = y_tensor[tr_idx], y_tensor[val_idx]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)

            model = get_model(
                model_name,
                input_size=X_tr.shape[2],
                output_size=2,
                device=device,
                seconds_min=seconds_min,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=15)

            save_path = os.path.join(model_dir, f"{model_name}_fold{fold}_best.pth")

            model, history = train_loop(
                model,
                train_loader,
                val_loader,
                nn.CrossEntropyLoss(),
                optimizer,
                scheduler,
                device,
                model_name,
                epochs=500,
                patience=15,
                save_path=save_path,
                plot_path=f"{output_name}_fold_{fold}",
            )

            # Métricas del fold
            acc, prec, rec, f1 = evaluate_metrics(model, X_val, y_val, device, model_name)
            for k, v in zip(fold_metrics.keys(), [acc, prec, rec, f1]):
                fold_metrics[k].append(v)

            # Selección del mejor modelo del CV (mismo criterio que tu script original)
            if history["val_acc"][-1] > best_val_acc:
                best_val_acc = history["val_acc"][-1]
                best_model_path = save_path

        # ----------- Resumen CV -----------
        def stat(name: str, vals: List[float]) -> str:
            mu, sig = np.mean(vals), np.std(vals)
            cv = sig / mu * 100 if mu else 0
            return f"{name}: media={mu:.4f} | std={sig:.4f} | cv={cv:.2f}%"

        print("\nResultados K-Fold (CV determinista por persona)")
        for metric, vals in fold_metrics.items():
            print(stat(metric, vals))

        with open(f"{output_name}_kfold_metrics.txt", "w", encoding="utf-8") as fh:
            fh.write("Resultados K-Fold (CV determinista por persona)\n")
            fh.write(f"CV_SEED={CV_SEED} | K={CV_K}\n\n")
            for m, v in fold_metrics.items():
                fh.write(stat(m, v) + "\n")

        # ----------- Evaluación final (blind test: 60 reales) -----------
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

        print(f"\nEvaluación en BLIND TEST ({len(real_val_names)} audios reales no vistos)…")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device).eval()
        probs, preds, labels = validate_real_data(model, test_loader, device, model_name)

        # Guardar predicciones por muestra
        df_pred = pd.DataFrame({"y_true": labels, "y_pred": preds, "y_prob": probs})
        df_pred.to_csv(f"{output_name}_blind_predictions.csv", index=False)

        # Confusion matrix + métricas derivadas
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity/recall
        tnr = tn / (tn + fp) if (tn + fp) else 0.0  # specificity
        ppv = tp / (tp + fp) if (tp + fp) else 0.0  # precision
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        bal_acc = 0.5 * (tpr + tnr)
        ap = average_precision_score(labels, probs)  # PR-AUC

        with open(f"{output_name}_blind_confusion_and_metrics.txt", "w", encoding="utf-8") as fh:
            fh.write(f"TP={tp}\nTN={tn}\nFP={fp}\nFN={fn}\n")
            fh.write(f"Sensitivity_TPR={tpr:.6f}\nSpecificity_TNR={tnr:.6f}\n")
            fh.write(f"Precision_PPV={ppv:.6f}\nNPV={npv:.6f}\n")
            fh.write(f"BalancedAcc={bal_acc:.6f}\nAP={ap:.6f}\n")

        # ---- Bootstrap 95% CI on blind test set ----
        ci = bootstrap_ci_binary(labels, preds, probs, n_boot=2000, seed=SEED_BLIND)

        acc_point = accuracy_score(labels, preds)
        f1_point = f1_score(labels, preds)
        auc_point = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else np.nan

        print(f"[BOOTSTRAP 95% CI] ACC={acc_point:.4f} [{ci['acc_ci'][0]:.4f}, {ci['acc_ci'][1]:.4f}]")
        print(f"[BOOTSTRAP 95% CI]  F1={f1_point:.4f} [{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]")
        print(f"[BOOTSTRAP 95% CI] AUC={auc_point:.4f} [{ci['auc_ci'][0]:.4f}, {ci['auc_ci'][1]:.4f}]")

        with open(f"{output_name}_blind_bootstrap_ci.txt", "w", encoding="utf-8") as fh:
            fh.write(f"ACC={acc_point:.6f} CI95=[{ci['acc_ci'][0]:.6f}, {ci['acc_ci'][1]:.6f}]\n")
            fh.write(f"F1 ={f1_point:.6f} CI95=[{ci['f1_ci'][0]:.6f}, {ci['f1_ci'][1]:.6f}]\n")
            fh.write(f"AUC={auc_point:.6f} CI95=[{ci['auc_ci'][0]:.6f}, {ci['auc_ci'][1]:.6f}]\n")

        plot_classification_report(labels, preds, ["Control", "Patológico"], output_name)
        plot_confusion_matrix(labels, preds, ["Control", "Patológico"], output_name)
        plot_roc_curve(labels, probs, output_name)

print("\nEntrenamiento completado")
