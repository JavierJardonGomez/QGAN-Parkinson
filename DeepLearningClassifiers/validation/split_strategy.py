from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

def hold_out_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def k_fold_split(X, y, k=5):
    return KFold(n_splits=k, shuffle=True, random_state=42).split(X, y)

def leave_one_out_split(X, y):
    return LeaveOneOut().split(X, y)
