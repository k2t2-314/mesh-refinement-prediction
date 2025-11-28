import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

# 1) Load both part-level tables
df_part = pd.read_csv("part_level.csv", dtype={"part_id": str})
df_part = df_part.loc[:, ~df_part.columns.str.contains('^Unnamed')]

df_feature = pd.read_csv("feature_level_augmented.csv", dtype={"part_id": str})
df_feature = df_feature.loc[:, ~df_feature.columns.str.contains('^Unnamed')]

# 2) Split into train / val / test sets
unique_parts = df_part["part_id"].unique()

# Unified split based on part_id
train_parts, temp_parts = train_test_split(unique_parts, test_size=0.20, random_state=42)
val_parts, test_parts   = train_test_split(temp_parts,   test_size=0.50, random_state=42)

# Apply to part-level table
df_train_p = df_part[df_part["part_id"].isin(train_parts)]
df_val_p   = df_part[df_part["part_id"].isin(val_parts)]
df_test_p  = df_part[df_part["part_id"].isin(test_parts)]

# Apply to feature-level table
df_train_f = df_feature[df_feature["part_id"].isin(train_parts)]
df_val_f   = df_feature[df_feature["part_id"].isin(val_parts)]
df_test_f  = df_feature[df_feature["part_id"].isin(test_parts)]


# 4) Model 1: the feature needs refine or not —— MLP
# Preprocess
drop_cols = [
    "part_id",
    "refine_needed",
    "converged",
    "local_size",
    "global_size_initial", "global_size_final", "load_id"
]

X_train1 = df_train_f.drop(columns=drop_cols)
X_val1   = df_val_f.drop(columns=drop_cols)
X_test1  = df_test_f.drop(columns=drop_cols)

y_train1 = df_train_f["refine_needed"].astype(float)
y_val1   = df_val_f["refine_needed"].astype(float)
y_test1  = df_test_f["refine_needed"].astype(float)

numeric_cols1 = X_train1.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols1 = X_train1.select_dtypes(include=["object"]).columns.tolist()

preprocessor1 = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols1),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols1)
    ]
)

X_train_p1 = preprocessor1.fit_transform(X_train1)
X_val_p1   = preprocessor1.transform(X_val1)
X_test_p1  = preprocessor1.transform(X_test1)

X_train_t1 = torch.tensor(X_train_p1.toarray() if hasattr(X_train_p1, "toarray") else X_train_p1, dtype=torch.float32)
X_val_t1   = torch.tensor(X_val_p1.toarray() if hasattr(X_val_p1, "toarray") else X_val_p1, dtype=torch.float32)
X_test_t1  = torch.tensor(X_test_p1.toarray() if hasattr(X_test_p1, "toarray") else X_test_p1, dtype=torch.float32)

y_train_t1 = torch.tensor(y_train1.values, dtype=torch.float32).unsqueeze(1)
y_val_t1   = torch.tensor(y_val1.values,   dtype=torch.float32).unsqueeze(1)
y_test_t1  = torch.tensor(y_test1.values,  dtype=torch.float32).unsqueeze(1)

class RefineClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


model = RefineClassifier(X_train_t1.shape[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_t, y_train_t = X_train_t1.to(device), y_train_t1.to(device)
X_val_t,   y_val_t   = X_val_t1.to(device),   y_val_t1.to(device)
X_test_t,  y_test1_t  = X_test_t1.to(device),  y_test_t1.to(device)

pos_weight = 1.2*(y_train_t == 0).sum() / (y_train_t == 1).sum()
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

best_val_loss = float('inf')
best_model_state = None
patience = 10
counter = 0

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    pred = model(X_train_t)
    loss = bce(pred, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = bce(val_pred, y_val_t)
            val_acc = ((torch.sigmoid(val_pred) > 0.65) == y_val_t).float().mean()
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}")

        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    val_logits = model(X_val_t)
    val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
    y_val_np = y_val_t.cpu().numpy().flatten()

    fpr, tpr, thresholds = roc_curve(y_val_np, val_probs)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_thresh = thresholds[best_idx]

    print(f"\n[Threshold Selection by Youden's J]")
    print(f"Best Threshold = {best_thresh:.4f}")
    print(f"TPR = {tpr[best_idx]:.4f}, FPR = {fpr[best_idx]:.4f}, J = {j_scores[best_idx]:.4f}")



with torch.no_grad():
    test_pred_logits = model(X_test_t)
    test_pred_probs = torch.sigmoid(test_pred_logits)
    test_pred_labels = (test_pred_probs > best_thresh).int()
    test_acc = (test_pred_labels == y_test1_t.int()).float().mean()
    print(f"Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_test1_t.cpu().numpy(), test_pred_labels.cpu().numpy())
    print("Confusion Matrix:")
    print(cm)

print(classification_report(
    y_test1_t.cpu().numpy(),
    test_pred_labels.cpu().numpy(),
    digits=4
))

os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/model1.pt")
joblib.dump(preprocessor1, "saved_model/preprocessor1.pkl")
# joblib.dump(best_thresh, "saved_model/best_thresh1.pkl")

# 5) Model 2: the model converged or not
# Preprocess
feature_cols = [
    col for col in df_feature.columns
    if col not in ["part_id", "load_id", "refine_needed", "converged", "local_size", "global_size_initial", "global_size_final"]
]
part_level_cols = [
    col for col in df_part.columns
    if col not in ["part_id", "converged", "load_id", "top2_features", "global_size_initial", "global_size_final", "local_refine_needed", "local_size"]
]
label_col = "converged"

def extract_supervised_features(part_id, load_id):
    part_row = df_part[(df_part.part_id == part_id) & (df_part.load_id == load_id)].iloc[0]
    feat_rows = df_feature[(df_feature.part_id == part_id) & (df_feature.load_id == load_id)]

    # 1. fixed
    fixed_row = feat_rows[feat_rows.role == "fixed"]
    if len(fixed_row):
        X_fixed = preprocessor1.transform(fixed_row[feature_cols])
        with torch.no_grad():
            fixed_prob = torch.sigmoid(model(torch.tensor(X_fixed, dtype=torch.float32, device=device))).item()
        fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
    else:
        fixed_feats = [0] * (len(feature_cols) + 1)

    # 2. load
    load_row = feat_rows[feat_rows.role == "load"]
    if len(load_row):
        X_load = preprocessor1.transform(load_row[feature_cols])
        with torch.no_grad():
            load_prob = torch.sigmoid(model(torch.tensor(X_load, dtype=torch.float32, device=device))).item()
        load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
    else:
        load_feats = [0] * (len(feature_cols) + 1)

    # 3. free
    free_rows = feat_rows[feat_rows.role == "free"]
    if len(free_rows):
        X_free = preprocessor1.transform(free_rows[feature_cols])
        with torch.no_grad():
            free_probs = torch.sigmoid(model(torch.tensor(X_free, dtype=torch.float32, device=device))).cpu().numpy().flatten()
        best_idx = np.argmax(free_probs)
        free_feats = list(free_rows.iloc[best_idx][feature_cols]) + [free_probs[best_idx]]
    else:
        free_feats = [0] * (len(feature_cols) + 1)

    global_feats = list(part_row[part_level_cols])
    y = int(part_row[label_col])

    return fixed_feats + load_feats + free_feats + global_feats + [y]

rows = []
for pid, lid in df_part[["part_id", "load_id"]].drop_duplicates().values:
    rows.append([pid] + extract_supervised_features(pid, lid))

col_names = (
    ["part_id"] +
    [f"fixed_{c}" for c in feature_cols] + ["fixed_refine_prob"] +
    [f"load_{c}" for c in feature_cols] + ["load_refine_prob"] +
    [f"free_{c}" for c in feature_cols] + ["free_refine_prob"] +
    part_level_cols + ["label"]
)
df_sup = pd.DataFrame(rows, columns=col_names)

class ConvergeClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

groups = df_sup["part_id"]
X2 = df_sup.drop(columns=["part_id", "label"])
y2 = df_sup["label"].astype(int)

gkf = GroupKFold(n_splits=5)
fold_reports = []
fold_confusions = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X2, y2, groups)):
    print(f"Fold {fold+1}")
    X_train, X_test = X2.iloc[train_idx], X2.iloc[test_idx]
    y_train, y_test = y2.iloc[train_idx], y2.iloc[test_idx]

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    num_cols = [c for c in num_cols if not c.startswith("Unnamed")]
    cat_cols = [c for c in cat_cols if not c.startswith("Unnamed")]

    X_train[num_cols] = X_train[num_cols].fillna(0)
    X_test[num_cols]  = X_test[num_cols].fillna(0)
    for cat in cat_cols:
        X_train[cat] = X_train[cat].fillna("missing")
        X_test[cat]  = X_test[cat].fillna("missing")

    from sklearn.feature_selection import VarianceThreshold
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_train[num_cols])
    good_num_idx = vt.get_support(indices=True)
    num_cols = [num_cols[i] for i in good_num_idx]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    X_train_t = torch.tensor(X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test_p.toarray()  if hasattr(X_test_p, "toarray")  else X_test_p,  dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long, device=device)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.long, device=device)

    model_cv = ConvergeClassifier(X_train_t.shape[1]).to(device)
    optimizer_cv = optim.Adam(model_cv.parameters(), lr=1e-4, weight_decay=1e-3)
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    counter = 0

    # compute class weights for this fold
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    criterion_cv = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    for epoch in range(500):
        model_cv.train()
        optimizer_cv.zero_grad()
        pred_cv = model_cv(X_train_t)
        loss_cv = criterion_cv(pred_cv, y_train_t)
        loss_cv.backward()
        optimizer_cv.step()
        if epoch % 10 == 0:
            model_cv.eval()
            with torch.no_grad():
                val_logits = model_cv(X_test_t)
                val_loss = criterion_cv(val_logits, y_test_t)
                val_acc = (val_logits.argmax(1) == y_test_t).float().mean()
                print(f"  [Fold {fold+1}] Epoch {epoch:03d} | Train Loss: {loss_cv.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}")
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model_cv.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
    if best_model_state is not None:
        model_cv.load_state_dict(best_model_state)

    model_cv.eval()
    with torch.no_grad():
        logits = model_cv(X_test_t)
        preds = logits.argmax(1).cpu().numpy()
        print(classification_report(y_test_t.cpu().numpy(), preds, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test_t.cpu().numpy(), preds))
        fold_reports.append(classification_report(y_test_t.cpu().numpy(), preds, digits=4, output_dict=True))
        fold_confusions.append(confusion_matrix(y_test_t.cpu().numpy(), preds))

print("Cross-validation Results")
all_acc = [r['accuracy'] for r in fold_reports]
print(f"Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")

X_all = df_sup.drop(columns=["part_id", "label"])
y_all = df_sup["label"].astype(int)

num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_all.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in num_cols if not c.startswith("Unnamed")]
cat_cols = [c for c in cat_cols if not c.startswith("Unnamed")]

X_all[num_cols] = X_all[num_cols].fillna(0)
for cat in cat_cols:
    X_all[cat] = X_all[cat].fillna("missing")

vt = VarianceThreshold(threshold=0.0)
vt.fit(X_all[num_cols])
good_num_idx = vt.get_support(indices=True)
num_cols = [num_cols[i] for i in good_num_idx]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
X_all_p = preprocessor.fit_transform(X_all)
X_all_t = torch.tensor(X_all_p.toarray() if hasattr(X_all_p, "toarray") else X_all_p, dtype=torch.float32, device=device)
y_all_t = torch.tensor(y_all.values, dtype=torch.long, device=device)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_all
)
criterion_final = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
)

final_model = ConvergeClassifier(X_all_t.shape[1]).to(device)
optimizer_final = optim.Adam(final_model.parameters(), lr=1e-4, weight_decay=1e-3)
best_val_loss = float('inf')
best_model_state = None
patience = 10
counter = 0

for epoch in range(500):
    final_model.train()
    optimizer_final.zero_grad()
    pred = final_model(X_all_t)
    loss = criterion_final(pred, y_all_t)
    loss.backward()
    optimizer_final.step()
    if loss.item() < best_val_loss:
        best_val_loss = loss.item()
        best_model_state = final_model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

if best_model_state is not None:
    final_model.load_state_dict(best_model_state)

final_model.eval()
with torch.no_grad():
    logits = final_model(X_all_t)
    preds = logits.argmax(1).cpu().numpy()
    print("[Final Model on All Data]")
    print(classification_report(y_all_t.cpu().numpy(), preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_all_t.cpu().numpy(), preds))

os.makedirs("saved_model", exist_ok=True)
torch.save(final_model.state_dict(), "saved_model/model2.pt")
joblib.dump(preprocessor, "saved_model/preprocessor2.pkl")

# 6) Model 3: local refinement size
# Preprocess
drop_cols3 = [
    "part_id",
    "refine_needed",
    "converged",
    "local_size",
    "global_size_initial", "global_size_final", "load_id"
]

X_train3 = df_train_f.drop(columns=drop_cols3)
X_val3   = df_val_f.drop(columns=drop_cols3)
X_test3  = df_test_f.drop(columns=drop_cols3)

y_train3 = df_train_f["local_size"].astype(float)
y_val3   = df_val_f["local_size"].astype(float)
y_test3  = df_test_f["local_size"].astype(float)

numeric_cols3 = X_train3.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols3 = X_train3.select_dtypes(include=["object"]).columns.tolist()

preprocessor3 = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols3),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols3)
    ]
)

X_train_p3 = preprocessor3.fit_transform(X_train3)
X_val_p3   = preprocessor3.transform(X_val3)
X_test_p3  = preprocessor3.transform(X_test3)

X_train_t3 = torch.tensor(X_train_p3.toarray() if hasattr(X_train_p3, "toarray") else X_train_p3, dtype=torch.float32)
X_val_t3   = torch.tensor(X_val_p3.toarray() if hasattr(X_val_p3, "toarray") else X_val_p3, dtype=torch.float32)
X_test_t3  = torch.tensor(X_test_p3.toarray() if hasattr(X_test_p3, "toarray") else X_test_p3, dtype=torch.float32)

y_train_t3 = torch.tensor(y_train3.values, dtype=torch.float32).unsqueeze(1)
y_val_t3   = torch.tensor(y_val3.values,   dtype=torch.float32).unsqueeze(1)
y_test_t3  = torch.tensor(y_test3.values,  dtype=torch.float32).unsqueeze(1)

class LocalRefineRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)

model3 = LocalRefineRegressor(X_train_t3.shape[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model3 = model3.to(device)
X_train_t3, y_train_t3 = X_train_t3.to(device), y_train_t3.to(device)
X_val_t3,   y_val_t3   = X_val_t3.to(device),   y_val_t3.to(device)
X_test_t3,  y_test_t3  = X_test_t3.to(device),  y_test_t3.to(device)

mse3 = nn.MSELoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-4, weight_decay=1e-3)

best_val_loss3 = float('inf')
best_model_state3 = None
patience3 = 10
counter3 = 0

for epoch in range(500):
    model3.train()
    optimizer3.zero_grad()
    pred3 = model3(X_train_t3)
    loss3 = mse3(pred3, y_train_t3)
    loss3.backward()
    optimizer3.step()

    if epoch % 10 == 0:
        model3.eval()
        with torch.no_grad():
            val_pred3 = model3(X_val_t3)
            val_loss3 = mse3(val_pred3, y_val_t3)
            print(f"[Model 3] Epoch {epoch:03d} | Train Loss: {loss3.item():.4f} | Val Loss: {val_loss3.item():.4f}")

        if val_loss3.item() < best_val_loss3:
            best_val_loss3 = val_loss3.item()
            best_model_state3 = model3.state_dict()
            counter3 = 0
        else:
            counter3 += 1
            if counter3 >= patience3:
                print(f"[Model 3] Early stopping at epoch {epoch}")
                break

if best_model_state3 is not None:
    model3.load_state_dict(best_model_state3)

model3.eval()
with torch.no_grad():
    test_pred3 = model3(X_test_t3)
    test_mse3 = mse3(test_pred3, y_test_t3)
    test_mae3 = torch.mean(torch.abs(test_pred3 - y_test_t3))
    print(f"[Model 3] Test MSE: {test_mse3.item():.4f} | Test MAE: {test_mae3.item():.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(model3.state_dict(), "saved_model/model3.pt")
joblib.dump(preprocessor3, "saved_model/preprocessor3.pkl")

# 7) Model 4: global refinement size
label_col_4 = "global_size_final"

def extract_supervised_features_4(part_id, load_id):
    part_row = df_part[(df_part.part_id == part_id) & (df_part.load_id == load_id)].iloc[0]
    feat_rows = df_feature[(df_feature.part_id == part_id) & (df_feature.load_id == load_id)]

    # ---------- fixed ----------
    fixed_row = feat_rows[feat_rows.role == "fixed"]
    if len(fixed_row):
        X_fixed = preprocessor1.transform(fixed_row[feature_cols])
        with torch.no_grad():
            fixed_prob = torch.sigmoid(
                model(torch.tensor(X_fixed, dtype=torch.float32, device=device))
            ).item()
        fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
    else:
        fixed_feats = [0] * (len(feature_cols) + 1)

    # ---------- load ----------
    load_row = feat_rows[feat_rows.role == "load"]
    if len(load_row):
        X_load = preprocessor1.transform(load_row[feature_cols])
        with torch.no_grad():
            load_prob = torch.sigmoid(
                model(torch.tensor(X_load, dtype=torch.float32, device=device))
            ).item()
        load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
    else:
        load_feats = [0] * (len(feature_cols) + 1)

    # ---------- free ----------
    free_rows = feat_rows[feat_rows.role == "free"]
    if len(free_rows):
        X_free = preprocessor1.transform(free_rows[feature_cols])
        with torch.no_grad():
            free_probs = torch.sigmoid(
                model(torch.tensor(X_free, dtype=torch.float32, device=device))
            ).cpu().numpy().flatten()
        best_idx = np.argmax(free_probs)
        free_feats = list(free_rows.iloc[best_idx][feature_cols]) + [free_probs[best_idx]]
    else:
        free_feats = [0] * (len(feature_cols) + 1)

    # ---------- global ----------
    global_feats = list(part_row[part_level_cols])
    y = float(part_row[label_col_4])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Must include part_id for grouping
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
    return [part_id] + fixed_feats + load_feats + free_feats + global_feats + [y]


# ---- build supervised table ----
rows_4 = []
for pid, lid in df_part[["part_id", "load_id"]].drop_duplicates().values:
    rows_4.append(extract_supervised_features_4(pid, lid))

col_names_4 = (
    ["part_id"] +
    [f"fixed_{c}" for c in feature_cols] + ["fixed_refine_prob"] +
    [f"load_{c}" for c in feature_cols] + ["load_refine_prob"] +
    [f"free_{c}" for c in feature_cols] + ["free_refine_prob"] +
    part_level_cols + ["label"]
)

df_sup4 = pd.DataFrame(rows_4, columns=col_names_4)


# -------------------------------------
# Separate X/y and part groups
# -------------------------------------
X4 = df_sup4.drop(columns=["label", "part_id"])
y4 = df_sup4["label"].astype(float)
groups4 = df_sup4["part_id"]


############################################
# Model Definition
############################################
class GlobalSizeRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


############################################
# 5-fold GroupKFold CV
############################################
gkf4 = GroupKFold(n_splits=5)
fold_metrics4 = []

for fold, (train_idx, test_idx) in enumerate(gkf4.split(X4, y4, groups4)):
    print(f"\n[Model 4] Fold {fold+1}")

    X_train4 = X4.iloc[train_idx].copy()
    X_test4  = X4.iloc[test_idx].copy()
    y_train4 = y4.iloc[train_idx].values
    y_test4  = y4.iloc[test_idx].values

    # ---- preprocess ----
    num_cols4 = X_train4.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols4 = X_train4.select_dtypes(include=["object"]).columns.tolist()

    X_train4[num_cols4] = X_train4[num_cols4].fillna(0)
    X_test4[num_cols4]  = X_test4[num_cols4].fillna(0)
    for col in cat_cols4:
        X_train4[col] = X_train4[col].fillna("missing")
        X_test4[col]  = X_test4[col].fillna("missing")

    # remove zero-variance
    vt4 = VarianceThreshold(threshold=0.0)
    vt4.fit(X_train4[num_cols4])
    good_idx = vt4.get_support(indices=True)
    num_cols4 = [num_cols4[i] for i in good_idx]

    preprocessor4 = ColumnTransformer([
        ("num", StandardScaler(), num_cols4),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols4)
    ])

    X_train4_p = preprocessor4.fit_transform(X_train4)
    X_test4_p  = preprocessor4.transform(X_test4)

    X_train4_t = torch.tensor(
        X_train4_p.toarray() if hasattr(X_train4_p, "toarray") else X_train4_p,
        dtype=torch.float32, device=device
    )
    X_test4_t = torch.tensor(
        X_test4_p.toarray() if hasattr(X_test4_p, "toarray") else X_test4_p,
        dtype=torch.float32, device=device
    )
    y_train4_t = torch.tensor(y_train4, dtype=torch.float32, device=device).unsqueeze(1)
    y_test4_t  = torch.tensor(y_test4, dtype=torch.float32, device=device).unsqueeze(1)

    # ---- model ----
    model4_cv = GlobalSizeRegressor(X_train4_t.shape[1]).to(device)
    optimizer4_cv = optim.Adam(model4_cv.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion4 = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    patience = 10
    counter = 0

    for epoch in range(500):
        model4_cv.train()
        optimizer4_cv.zero_grad()
        pred = model4_cv(X_train4_t)
        loss = criterion4(pred, y_train4_t)
        loss.backward()
        optimizer4_cv.step()

        if epoch % 20 == 0:
            model4_cv.eval()
            with torch.no_grad():
                val_pred = model4_cv(X_test4_t)
                val_loss = criterion4(val_pred, y_test4_t)
                val_mae = torch.mean(torch.abs(val_pred - y_test4_t)).item()

            print(f"[Model 4] Fold {fold+1} Epoch {epoch:03d} "
                  f"| Train Loss: {loss.item():.4f} "
                  f"| Val Loss: {val_loss.item():.4f} "
                  f"| Val MAE: {val_mae:.4f}")

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_state = model4_cv.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"[Model 4] Early stopping at epoch {epoch}")
                    break

    if best_state:
        model4_cv.load_state_dict(best_state)

    model4_cv.eval()
    with torch.no_grad():
        pred = model4_cv(X_test4_t).cpu().numpy().flatten()
        mse = np.mean((pred - y_test4)**2)
        mae = np.mean(np.abs(pred - y_test4))
        fold_metrics4.append((mse, mae))

        print(f"[Model 4] Fold {fold+1} MSE: {mse:.4f} | MAE: {mae:.4f}")


print("Model 4 Cross-validation Results")
mse_all = [m[0] for m in fold_metrics4]
mae_all = [m[1] for m in fold_metrics4]
print(f"Mean MSE: {np.mean(mse_all):.4f} ± {np.std(mse_all):.4f}")
print(f"Mean MAE: {np.mean(mae_all):.4f} ± {np.std(mae_all):.4f}")


############################################
# Train Final Model on ALL Data
############################################
X4_all = X4.copy()
y4_all = y4.copy()

num_cols_all4 = X4_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all4 = X4_all.select_dtypes(include=["object"]).columns.tolist()

X4_all[num_cols_all4] = X4_all[num_cols_all4].fillna(0)
for col in cat_cols_all4:
    X4_all[col] = X4_all[col].fillna("missing")

vt4_all = VarianceThreshold(threshold=0.0)
vt4_all.fit(X4_all[num_cols_all4])
good_idx = vt4_all.get_support(indices=True)
num_cols_all4 = [num_cols_all4[i] for i in good_idx]

preprocessor4_all = ColumnTransformer([
    ("num", StandardScaler(), num_cols_all4),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_all4)
])
X4_all_p = preprocessor4_all.fit_transform(X4_all)

X4_all_t = torch.tensor(
    X4_all_p.toarray() if hasattr(X4_all_p, "toarray") else X4_all_p,
    dtype=torch.float32, device=device
)
y4_all_t = torch.tensor(y4_all.values, dtype=torch.float32, device=device).unsqueeze(1)

final_model4 = GlobalSizeRegressor(X4_all_t.shape[1]).to(device)
optimizer4_final = optim.Adam(final_model4.parameters(), lr=1e-4, weight_decay=1e-3)
criterion4_final = nn.MSELoss()

best_loss4 = float('inf')
best_state4 = None
counter = 0
patience = 10

for epoch in range(500):
    final_model4.train()
    optimizer4_final.zero_grad()
    pred = final_model4(X4_all_t)
    loss = criterion4_final(pred, y4_all_t)
    loss.backward()
    optimizer4_final.step()

    if loss.item() < best_loss4:
        best_loss4 = loss.item()
        best_state4 = final_model4.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"[Model 4] Early stopping at epoch {epoch}")
            break

if best_state4:
    final_model4.load_state_dict(best_state4)

final_model4.eval()
with torch.no_grad():
    pred_all = final_model4(X4_all_t).cpu().numpy().flatten()
    mse_all_final = np.mean((pred_all - y4_all.values)**2)
    mae_all_final = np.mean(np.abs(pred_all - y4_all.values))

print("\n[Model 4 Final Model on ALL Data]")
print(f"MSE: {mse_all_final:.4f} | MAE: {mae_all_final:.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(final_model4.state_dict(), "saved_model/model4.pt")
joblib.dump(preprocessor4_all, "saved_model/preprocessor4.pkl")
