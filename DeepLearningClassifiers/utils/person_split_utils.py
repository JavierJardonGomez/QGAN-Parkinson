import os
import random
import re
from pathlib import Path
from typing import Iterator

import numpy as np


def strip_suffix_label(stem: str) -> str:
    """Quita _c / _p al final del stem (si existe)."""
    return re.sub(r"_[cp]$", "", stem)


def base_id(fname: str) -> str:
    """
    Devuelve identificador base sin sufijos de augmentacion/generacion ni label final.
    Importante para que real y sintetico compartan la misma "base".
    """
    stem = Path(fname).stem
    stem = re.sub(r"_aug\d+", "", stem)
    stem = re.sub(r"_generated$", "", stem)
    stem = strip_suffix_label(stem)
    return stem


def person_id(fname: str) -> str:
    """
    Devuelve id estable de persona.
    Ej: AVPEPUDEA0001a1_p.wav -> AVPEPUDEA0001
        AVPEPUDEAC0005a3_c.wav -> AVPEPUDEAC0005

    Asume patron ...####<letra><digito> (p.ej. 0001a1, 0001a2, 0001a3).
    """
    stem = base_id(fname)

    match = re.match(r"^(.*?\d{4})[A-Za-z]\d$", stem)
    if not match:
        return stem
    return match.group(1)


def select_people_with_3_variants(
    files: list[str], n_people: int, rng: random.Random
) -> tuple[list[str], list[str]]:
    """
    Selecciona n_people personas (con >=3 tomas reales) y devuelve:
      - personas elegidas
      - exactamente 3 ficheros por persona (ordenados)
    """
    by_person: dict[str, list[str]] = {}
    for file_name in files:
        pid = person_id(file_name)
        by_person.setdefault(pid, []).append(file_name)

    eligible = [pid for pid, file_list in by_person.items() if len(file_list) >= 3]
    if len(eligible) < n_people:
        raise ValueError(
            f"No hay suficientes personas elegibles con >=3 variantes: {len(eligible)} < {n_people}"
        )

    chosen_pids = rng.sample(sorted(eligible), n_people)

    chosen_files: list[str] = []
    for pid in chosen_pids:
        chosen_files.extend(sorted(by_person[pid])[:3])

    return chosen_pids, chosen_files


def k_fold_split_by_person_seeded(
    groups: np.ndarray, k: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold determinista: baraja personas con seed y crea folds sobre grupos.
    Devuelve indices (tr_idx, val_idx) sobre el dataset.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    folds = np.array_split(unique_groups, k)
    for i in range(k):
        val_groups = set(folds[i].tolist())
        val_mask = np.array([group in val_groups for group in groups], dtype=bool)
        val_idx = np.where(val_mask)[0]
        tr_idx = np.where(~val_mask)[0]
        yield tr_idx, val_idx


def write_blind_log(
    exp_dir: str,
    seed_blind: int,
    n_people_per_class: int,
    chosen_ctrl_pids: list[str],
    chosen_pat_pids: list[str],
    ctrl_val_files: list[str],
    pat_val_files: list[str],
    real_val_names: list[str],
) -> None:
    log_path = os.path.join(exp_dir, "blind_test_selection_log.txt")
    with open(log_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"SEED_BLIND={seed_blind}\n")
        file_handle.write(
            f"N_PEOPLE_PER_CLASS={n_people_per_class} (total_people={2 * n_people_per_class})\n"
        )
        file_handle.write(
            f"Total blind audios={len(real_val_names)} (expected={2 * n_people_per_class * 3})\n\n"
        )

        file_handle.write("CONTROL people:\n")
        for pid in chosen_ctrl_pids:
            file_handle.write(f"  {pid}\n")
        file_handle.write("\nCONTROL files:\n")
        for file_name in sorted(ctrl_val_files):
            file_handle.write(f"  {file_name}\n")

        file_handle.write("\nPATOLOGICO people:\n")
        for pid in chosen_pat_pids:
            file_handle.write(f"  {pid}\n")
        file_handle.write("\nPATOLOGICO files:\n")
        for file_name in sorted(pat_val_files):
            file_handle.write(f"  {file_name}\n")


def write_cv_folds_log(groups: np.ndarray, exp_dir: str, k: int, seed: int) -> None:
    fold_log = os.path.join(exp_dir, "cv_folds_log.txt")
    with open(fold_log, "w", encoding="utf-8") as file_handle:
        file_handle.write(f"CV_SEED={seed}\n")
        file_handle.write(f"K={k}\n")
        file_handle.write(f"Total_samples={len(groups)}\n")
        file_handle.write(f"Total_people={len(np.unique(groups))}\n\n")

        all_people = set(np.unique(groups))

        for fold, (_tr_idx, val_idx) in enumerate(
            k_fold_split_by_person_seeded(groups, k=k, seed=seed), start=1
        ):
            val_pids = set(groups[val_idx])
            train_pids = all_people - val_pids

            file_handle.write(f"========== FOLD {fold} ==========\n")
            file_handle.write(f"VAL people ({len(val_pids)}):\n")
            for pid in sorted(val_pids):
                file_handle.write(f"  {pid}\n")

            file_handle.write(f"\nTRAIN people ({len(train_pids)}):\n")
            for pid in sorted(train_pids):
                file_handle.write(f"  {pid}\n")
            file_handle.write("\n\n")
