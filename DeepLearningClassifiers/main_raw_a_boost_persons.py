#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
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
from train.training_utils import train_loop, validate_real_data, evaluate_metrics, bootstrap_ci_binary
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
from models.inceptiontime_model import InceptionTimeEnsemble


# --------------------------- Config ------------------------------------------
vowel = "A"
AUDIO_DIR = f"/home/jovyan/QVocoder/paper/data/generated_vowels_aug/{vowel}"
seconds_min = 0.49

experiment_ids = [40, 41, 42, 43]
batch_sizes    = [8, 16, 32, 64]
lrs            = [1e-5, 2e-5, 5e-5, 8e-5]
weight_decays  = [1e-4, 1e-4, 1e-4, 1e-4]

TARGET_LEN = int(seconds_min * 22_050)
all_model_names = ["resnet", "inception", "conv", "lstm_fcn", "times_net"]
BASE_RESULTS_DIR = "paper_vowel_a_croped_resample_aug_boost_persons"

# Blind test: 20 personas (10C + 10P) * 3 audios = 60 audios
SEED_BLIND = 42
N_PEOPLE_PER_CLASS = 10

# CV: determinista por persona
CV_K = 5
CV_SEED = SEED_BLIND


# --------------------------- Dispositivo -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}\n")

# --------------------------- Main loop ---------------------------------------
for exp_id, bs, lr, wd in zip(experiment_ids, batch_sizes, lrs, weight_decays):
    print(f"=== Experimento {exp_id} | BS={bs}, LR={lr}, WD={wd} ===")
    exp_dir = os.path.join(BASE_RESULTS_DIR, f"V5_{exp_id}_bs{bs}_lr{lr}_wd{wd}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1) Listas de audio ------------------------------------------------------
    audio_dir = f"{AUDIO_DIR}"
    real_files, gen_files, label_dict = read_audios_dir(audio_dir)

    # Solo reales (para blind test y para entrenar junto con sintéticos)
    real_clean = [f for f in real_files if "_generated" not in f]
    real_c = [f for f in real_clean if f.endswith("_c")]
    real_p = [f for f in real_clean if f.endswith("_p")]

    # 1.a) Blind test determinista por PERSONA (10C + 10P, 3 tomas/persona) ----
    rng = random.Random(SEED_BLIND)
    chosen_ctrl_pids, ctrl_val_files = select_people_with_3_variants(real_c, N_PEOPLE_PER_CLASS, rng)
    chosen_pat_pids,  pat_val_files  = select_people_with_3_variants(real_p, N_PEOPLE_PER_CLASS, rng)

    real_val_names = ctrl_val_files + pat_val_files  # 60 audios
    real_train_names = [f for f in real_clean if f not in set(real_val_names)]

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

    # 1.b) Filtrar sintéticos por PERSONA del blind test (evita fuga) ----------
    val_person_ids = {person_id(f) for f in real_val_names}

    gen_train = [
        f for f in gen_files
        if person_id(base_id(f)) not in val_person_ids
    ]

    train_names = real_train_names + gen_train

    # 2) Groups (por persona) -------------------------------------------------
    # Para CV: agrupa por persona para que a1/a2/a3 (y sus sintéticos) nunca se separen.
    groups = np.array([person_id(base_id(f)) for f in train_names])

    # Log de folds CV deterministas (trazabilidad)
    write_cv_folds_log(groups, exp_dir, k=CV_K, seed=CV_SEED)

    # 3) Features & labels ----------------------------------------------------
    X_train, y_train_labels = load_data_raw(train_names, label_dict, audio_dir, TARGET_LEN)
    X_test,  y_test_labels  = load_data_raw(real_val_names, label_dict, audio_dir, TARGET_LEN)

    enc = OneHotEncoder(sparse_output=False)
    y_onehot = enc.fit_transform(y_train_labels.reshape(-1, 1))

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_onehot, dtype=torch.float32)

    # 4) Cross-validation -----------------------------------------------------
    for model_name in all_model_names:
        print(f"\n--- Modelo: {model_name} ---")
        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        parsed_audio_dir = os.path.join(
            os.path.basename(os.path.dirname(AUDIO_DIR)),  # "generated_vowels_aug"
            os.path.basename(AUDIO_DIR)                    # "A"
        ).replace(os.sep, "_")
        out_base = os.path.join(model_dir, f"{parsed_audio_dir}_{exp_id}")
        print("out_base: ", out_base)

        best_acc, best_path = 0.0, ""
        metrics = {"acc": [], "prec": [], "rec": [], "f1": []}

        for fold, (tr_idx, va_idx) in enumerate(
            k_fold_split_by_person_seeded(groups, k=CV_K, seed=CV_SEED), start=1
        ):
            print(f"Fold {fold}: {len(tr_idx)} train - {len(va_idx)} val")
            X_tr, X_va = X_tensor[tr_idx], X_tensor[va_idx]
            y_tr, y_va = y_tensor[tr_idx], y_tensor[va_idx]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
            val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=bs)

            if model_name == "inception":
                model = InceptionTimeEnsemble(n_models=5, input_channels=1, nb_classes=2).to(device)
            else:
                model = get_model(
                    model_name,
                    input_size=X_tr.shape[2],
                    output_size=2,
                    device=device,
                    seconds_min=seconds_min
                )

            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
            sched = ReduceLROnPlateau(opt, mode="min", factor=0.4, patience=15)
            save_fold = os.path.join(model_dir, f"{model_name}_fold{fold}.pth")

            model, hist = train_loop(
                model, train_loader, val_loader, nn.CrossEntropyLoss(),
                opt, sched, device, model_name,
                epochs=500, patience=15,
                save_path=save_fold,
                plot_path=f"{out_base}_fold{fold}.png",
            )

            acc, pr, rc, f1 = evaluate_metrics(model, X_va, y_va, device, model_name)
            for k, v in zip(metrics.keys(), [acc, pr, rc, f1]):
                metrics[k].append(v)

            if hist["val_acc"][-1] > best_acc:
                best_acc, best_path = hist["val_acc"][-1], save_fold

        # 5) Resumen CV -------------------------------------------------------
        def summarize(name: str, vals: List[float]) -> str:
            mu, sd = np.mean(vals), np.std(vals)
            cv = (sd / mu * 100.0) if mu else 0.0
            return f"{name}: {mu:.4f} ± {sd:.4f} (CV {cv:.2f}%)"

        print("\nResultados k-fold (determinista por persona)")
        for m, v in metrics.items():
            print(summarize(m, v))

        with open(f"{out_base}_kfold.txt", "w", encoding="utf-8") as fh:
            fh.write(f"CV_SEED={CV_SEED} | K={CV_K}\n\n")
            for m, v in metrics.items():
                fh.write(summarize(m, v) + "\n")

        # 6) Blind test externo (60 reales) -----------------------------------
        print(f"\nEvaluación BLIND TEST ({len(real_val_names)} reales)…")
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device).eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(
            enc.transform(y_test_labels.reshape(-1, 1)), dtype=torch.float32
        )
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=16)

        probs, preds, labels = validate_real_data(model, test_loader, device, model_name)

        # Guardar predicciones por muestra
        df_pred = pd.DataFrame({"y_true": labels, "y_pred": preds, "y_prob": probs})
        df_pred.to_csv(f"{out_base}_blind_predictions.csv", index=False)

        # Confusion matrix + métricas derivadas
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        bal_acc = 0.5 * (tpr + tnr)
        ap = average_precision_score(labels, probs)

        with open(f"{out_base}_blind_confusion_and_metrics.txt", "w", encoding="utf-8") as fh:
            fh.write(f"TP={tp}\nTN={tn}\nFP={fp}\nFN={fn}\n")
            fh.write(f"Sensitivity_TPR={tpr:.6f}\nSpecificity_TNR={tnr:.6f}\n")
            fh.write(f"Precision_PPV={ppv:.6f}\nNPV={npv:.6f}\n")
            fh.write(f"BalancedAcc={bal_acc:.6f}\nAP={ap:.6f}\n")

        # Bootstrap CI
        ci = bootstrap_ci_binary(labels, preds, probs, n_boot=2000, seed=SEED_BLIND)

        acc_point = accuracy_score(labels, preds)
        f1_point  = f1_score(labels, preds)
        auc_point = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else np.nan

        print(f"[BOOTSTRAP 95% CI] ACC={acc_point:.4f} [{ci['acc_ci'][0]:.4f}, {ci['acc_ci'][1]:.4f}]")
        print(f"[BOOTSTRAP 95% CI]  F1={f1_point:.4f} [{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]")
        print(f"[BOOTSTRAP 95% CI] AUC={auc_point:.4f} [{ci['auc_ci'][0]:.4f}, {ci['auc_ci'][1]:.4f}]")

        with open(f"{out_base}_blind_bootstrap_ci.txt", "w", encoding="utf-8") as fh:
            fh.write(f"ACC={acc_point:.6f} CI95=[{ci['acc_ci'][0]:.6f}, {ci['acc_ci'][1]:.6f}]\n")
            fh.write(f"F1 ={f1_point:.6f} CI95=[{ci['f1_ci'][0]:.6f}, {ci['f1_ci'][1]:.6f}]\n")
            fh.write(f"AUC={auc_point:.6f} CI95=[{ci['auc_ci'][0]:.6f}, {ci['auc_ci'][1]:.6f}]\n")

        plot_classification_report(labels, preds, ["Control", "Patológico"], out_base)
        plot_confusion_matrix(labels, preds, ["Control", "Patológico"], out_base)
        plot_roc_curve(labels, probs, out_base)

print("\nEntrenamiento completado (CV por persona + blind test por persona)")
