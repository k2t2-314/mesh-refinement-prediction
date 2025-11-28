from .step_feature_detection import load_step_model, load_step_unit, detect_faces, detect_circular_holes, detect_non_circular_holes, detect_fillets, mode_thickness, detect_all_edges, detect_all_faces
import os, json
import pandas as pd
import gmsh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
import joblib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_mesh(nodes, elements, title, highlight_faces=None, face_to_nodes=None, save_path="mesh.png"):
    # nodes: (N,3)
    # elements: [(etype, conn), ...], conn is 0-based node indices
    # highlight_faces: list of gmsh surface IDs to mark red
    # face_to_nodes: dict: surface_id -> list[node_indices]

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)

    # normal mesh (black)
    for (etype, conn) in elements:
        for e in conn:
            pts = nodes[e][:, :3]
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:,0], pts[:,1], pts[:,2], "k-", linewidth=0.4)

    if highlight_faces and face_to_nodes:
        for sf in highlight_faces:
            if sf not in face_to_nodes:
                continue
            face_nodes = face_to_nodes[sf]
            for fn in face_nodes:
                pts = nodes[fn][:, :3]
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:,0], pts[:,1], pts[:,2], "r-", linewidth=1.2)

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_result(step_path, thickness, load_face, load_direction, load_scale):
    def step_to_json(step_path, thickness, part_id=None):
        shape = load_step_model(step_path)
        part_id = part_id or os.path.splitext(os.path.basename(step_path))[0]
        circular_holes = detect_circular_holes(shape)
        fillets = detect_fillets(shape)
        faces = detect_all_faces(shape)
        print(f"Detected {len(faces)} faces")

        features = {
            "part_id": part_id,
            "thickness": thickness,
            "circular_holes": circular_holes,
            "fillets": fillets,
            "faces": faces
        }
        out_path ="output/example.json"
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)

    step_to_json(step_path, thickness)

    gmsh.initialize()
    gmsh.open(step_path)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surface_info = []
    for dim, tag in surfaces:
        mass_props = gmsh.model.occ.getMass(dim, tag)
        com = gmsh.model.occ.getCenterOfMass(dim, tag)  # (x, y, z)
        surface_info.append({
            "tag": tag,
            "center": com,
            "area": mass_props
        })

    with open("output/example.json") as f:
        features = json.load(f)
    faces = features["faces"]

    num_fillet = len(features["fillets"])
    num_hole = len(features["circular_holes"])

    num_faces = len(features["faces"])
    num_plane = num_faces - num_fillet - num_hole

    smallest_fillet_r = min((f["radius"] for f in features["fillets"]), default=None)
    smallest_hole_r = min((h["radius"] for h in features["circular_holes"]), default=None)

    print(f"num_fillet: {num_fillet}, num_hole: {num_hole}, num_plane: {num_plane}")
    print(f"smallest_fillet_r: {smallest_fillet_r}, smallest_hole_r: {smallest_hole_r}")

    def detect_maxsize(shape):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox) 
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        maxsize = max(size_x, size_y, size_z)
        return maxsize, (size_x, size_y, size_z)

    shape = load_step_model(step_path)
    maxsize, (sx, sy, sz) = detect_maxsize(shape)
    print("maxsize:", maxsize)

    def find_the_faces(step_path):
        shape = load_step_model(step_path)
        faces = detect_all_faces(shape)
        z_vals = [f["center"][2] for f in faces]
        x_vals = [f["center"][0] for f in faces]
        y_vals = [f["center"][1] for f in faces]

        idx_bottom = z_vals.index(min(z_vals))
        idx_left   = x_vals.index(min(x_vals))
        idx_front  = y_vals.index(min(y_vals)) 
        idx_back   = y_vals.index(max(y_vals))

        return {
            "bottom": faces[idx_bottom], 
            "left":   faces[idx_left], 
            "front":  faces[idx_front],
            "back":   faces[idx_back],
            "bottom_idx": idx_bottom,
            "left_idx": idx_left,
            "front_idx": idx_front,
            "back_idx": idx_back,
        }

    face_dict = find_the_faces(step_path)

    part_row = {
        "num_fillet": num_fillet,
        "num_hole": num_hole,
        "num_plane": num_plane,
        "smallest_fillet_r": smallest_fillet_r,
        "smallest_hole_r": smallest_hole_r,
        "maxsize": maxsize,
        "thickness": thickness,
        "load_face": load_face,
        "load_direction": load_direction,
        "load_scale": load_scale,
    }
    df_part = pd.DataFrame([part_row])

    face_type_dict = {}  # face_id → ftype
    for hole in features["circular_holes"]:
        for f in hole["faces"]:
            face_type_dict[f["face_id"]] = "CIRC_HOLE"
    for fillet in features["fillets"]:
        for f in fillet["faces"]:
            face_type_dict[f["face_id"]] = "FILLET"
    for f in features["faces"]:
        if f["face_id"] not in face_type_dict:
            face_type_dict[f["face_id"]] = "FACE"

    if load_face == "bend_bottom":
        load_face_data = face_dict["bottom"]
        fix_face_data = face_dict["left"]

    elif load_face == "tension":
        load_face_data = face_dict["back"]
        fix_face_data = face_dict["front"]


    load_face_center = np.array(load_face_data["center"])
    load_face_normal = np.array(load_face_data["normal"])
    fix_face_center  = np.array(fix_face_data["center"])
    fix_face_normal  = np.array(fix_face_data["normal"])
                                
    def get_face_bbox_dict(shape):
        bbox_dict = {}
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 0
        while exp.More():
            face_shape = exp.Current()
            face = TopoDS.Face_s(face_shape)
            bbox = Bnd_Box()
            BRepBndLib.Add_s(face, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            bbox_dx = xmax - xmin
            bbox_dy = ymax - ymin
            bbox_dz = zmax - zmin
            bbox_dict[face_id] = (bbox_dx, bbox_dy, bbox_dz)
            face_id += 1
            exp.Next()
        return bbox_dict

    shape = load_step_model(step_path)
    bbox_dict = get_face_bbox_dict(shape)

    rows1 = []
    for f in faces:
        row = {}
        row["face_id"] = f["face_id"]
        row["ftype"]   = face_type_dict[f["face_id"]]
        row["cx"], row["cy"], row["cz"] = f["center"]
        row["nx"], row["ny"], row["nz"] = f["normal"]
        row["area"] = f.get("area")
        if f["face_id"] == load_face_data["face_id"]:
            row["role"] = "load"
        elif f["face_id"] == fix_face_data["face_id"]:
            row["role"] = "fixed"
        else:
            row["role"] = "free"
        dx, dy, dz = bbox_dict[f["face_id"]]
        row["bbox_dx"] = dx
        row["bbox_dy"] = dy
        row["bbox_dz"] = dz

        fc = np.array(f["center"])
        row["dist_load"] = np.linalg.norm(fc - np.array(load_face_center))
        row["nlx"], row["nly"], row["nlz"] = load_face_normal
        row["dist_fixed"] = np.linalg.norm(fc - np.array(fix_face_center))
        row["nfx"], row["nfy"], row["nfz"] = fix_face_normal

        row["load_face"] = load_face
        row["load_direction"] = load_direction
        row["load_scale"] = load_scale
        row["maxsize"] = maxsize
        row["thickness"] = thickness
        rows1.append(row)
    df_feature = pd.DataFrame(rows1)

    # Model 1: feature refine_needed
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

    preprocessor1 = joblib.load("saved_model/preprocessor1.pkl")
    feature_cols1 = [c for c in df_feature.columns if c not in ["part_id", "refine_needed", "converged", "local_size", "global_size_initial", "global_size_final", "load_id"]]
    X1 = preprocessor1.transform(df_feature[feature_cols1])
    X1_t = torch.tensor(X1.toarray() if hasattr(X1, "toarray") else X1, dtype=torch.float32)

    model1 = RefineClassifier(X1_t.shape[1])
    state_dict = torch.load("saved_model/model1.pt", map_location="cpu")
    model1.load_state_dict(state_dict)
    model1.eval()

    with torch.no_grad():
        logits1 = model1(X1_t)
        probs1 = torch.sigmoid(logits1).cpu().numpy().flatten()
    df_feature["refine_prob"] = probs1

    top2 = df_feature.sort_values("refine_prob", ascending=False).head(2)
    df_feature.drop(columns=["refine_prob"], inplace=True)

    # Model 2: the model converged or not
    preprocessor2 = joblib.load("saved_model/preprocessor2.pkl")

    feature_cols = [
        col for col in df_feature.columns
        if col not in ["part_id", "load_id", "refine_needed", "converged", "local_size", "global_size_initial", "global_size_final"]
    ]
    part_level_cols = [
        col for col in df_part.columns
        if col not in ["part_id", "converged", "load_id", "top2_features", "global_size_initial", "global_size_final", "local_refine_needed", "local_size"]
    ]
    label_col = "converged"

    def extract_supervised_features():
        part_row = df_part.iloc[0]
        feat_rows = df_feature

        # fixed
        fixed_row = feat_rows[feat_rows.role == "fixed"]
        if len(fixed_row):
            X_fixed = preprocessor1.transform(fixed_row[feature_cols])
            with torch.no_grad():
                fixed_prob = torch.sigmoid(model1(torch.tensor(X_fixed, dtype=torch.float32))).item()
            fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
        else:
            fixed_feats = [0] * (len(feature_cols) + 1)

        # load
        load_row = feat_rows[feat_rows.role == "load"]
        if len(load_row):
            X_load = preprocessor1.transform(load_row[feature_cols])
            with torch.no_grad():
                load_prob = torch.sigmoid(model1(torch.tensor(X_load, dtype=torch.float32))).item()
            load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
        else:
            load_feats = [0] * (len(feature_cols) + 1)

        # free
        free_rows = feat_rows[feat_rows.role == "free"]
        if len(free_rows):
            X_free = preprocessor1.transform(free_rows[feature_cols])
            with torch.no_grad():
                free_probs = torch.sigmoid(model1(torch.tensor(X_free, dtype=torch.float32))).cpu().numpy().flatten()
            best_idx = np.argmax(free_probs)
            free_feats = list(free_rows.iloc[best_idx][feature_cols]) + [free_probs[best_idx]]
        else:
            free_feats = [0] * (len(feature_cols) + 1)

        global_feats = list(part_row[part_level_cols])
        return fixed_feats + load_feats + free_feats + global_feats

    col_names = (
        [f"fixed_{c}" for c in feature_cols] + ["fixed_refine_prob"] +
        [f"load_{c}" for c in feature_cols] + ["load_refine_prob"] +
        [f"free_{c}" for c in feature_cols] + ["free_refine_prob"] +
        part_level_cols
    )

    row = extract_supervised_features()
    df_sup = pd.DataFrame([row], columns=col_names)

    preprocessor2 = joblib.load("saved_model/preprocessor2.pkl")
    missing_cols = set(preprocessor2.feature_names_in_) - set(df_sup.columns)
    for c in missing_cols:
        df_sup[c] = 0.0
    df_sup = df_sup[preprocessor2.feature_names_in_]

    X2 = preprocessor2.transform(df_sup)
    X2_t = torch.tensor(X2.toarray() if hasattr(X2, "toarray") else X2, dtype=torch.float32)

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

    model2 = ConvergeClassifier(X2_t.shape[1])
    state_dict = torch.load("saved_model/model2.pt", map_location="cpu")
    model2.load_state_dict(state_dict)
    model2.eval()

    with torch.no_grad():
        logits2 = model2(X2_t)
        probs2 = torch.softmax(logits2, dim=1).cpu().numpy()
        preds2 = probs2.argmax(1)

    print("Model 2 (part-level converge) predictions:")
    print("Probability not converged:", probs2[:,0])
    print("Probability converged:", probs2[:,1])
    print("Predicted class (0: unconverged, 1: converged):", preds2)

    # Model 3: local size
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

    preprocessor3 = joblib.load("saved_model/preprocessor3.pkl")
    feature_cols3 = [c for c in df_feature.columns if c not in ["part_id", "refine_needed", "converged", "local_size", "global_size_initial", "global_size_final", "load_id"]]
    X3 = preprocessor3.transform(df_feature[feature_cols3])
    X3_t = torch.tensor(X3.toarray() if hasattr(X3, "toarray") else X3, dtype=torch.float32)

    model3 = LocalRefineRegressor(X3_t.shape[1])
    state_dict = torch.load("saved_model/model3.pt", map_location="cpu")
    model3.load_state_dict(state_dict)
    model3.eval()

    with torch.no_grad():
        logits1 = model1(X1_t)
        probs1 = torch.sigmoid(logits1).cpu().numpy().flatten()
    df_feature["refine_prob"] = probs1

    top2 = df_feature.sort_values("refine_prob", ascending=False).head(2)

    with torch.no_grad():
        local_size_pred = model3(X3_t).cpu().numpy().flatten()
    df_feature["predicted_local_size"] = local_size_pred

    top2_with_local_size = top2.copy()
    top2_with_local_size["predicted_local_size"] = df_feature.loc[top2.index, "predicted_local_size"]

    print("Top-2 faces needing refinement and their predicted local sizes:")
    print(top2_with_local_size[["face_id", "ftype", "predicted_local_size"]].to_string(index=False))
    df_feature.drop(columns=["refine_prob", "predicted_local_size"], inplace=True)

    # Model 4: global size
    preprocessor4 = joblib.load("saved_model/preprocessor4.pkl")

    feature_cols = [
        col for col in df_feature.columns
        if col not in ["part_id", "load_id", "refine_needed", "converged", "local_size", "global_size_initial", "global_size_final"]
    ]
    part_level_cols = [
        col for col in df_part.columns
        if col not in ["part_id", "converged", "load_id", "top2_features", "global_size_initial", "global_size_final", "local_refine_needed", "local_size"]
    ]

    def extract_supervised_features():
        part_row = df_part.iloc[0]
        feat_rows = df_feature

        # fixed
        fixed_row = feat_rows[feat_rows.role == "fixed"]
        if len(fixed_row):
            X_fixed = preprocessor1.transform(fixed_row[feature_cols])
            with torch.no_grad():
                fixed_prob = torch.sigmoid(model1(torch.tensor(X_fixed, dtype=torch.float32))).item()
            fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
        else:
            fixed_feats = [0] * (len(feature_cols) + 1)

        # load
        load_row = feat_rows[feat_rows.role == "load"]
        if len(load_row):
            X_load = preprocessor1.transform(load_row[feature_cols])
            with torch.no_grad():
                load_prob = torch.sigmoid(model1(torch.tensor(X_load, dtype=torch.float32))).item()
            load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
        else:
            load_feats = [0] * (len(feature_cols) + 1)

        # free
        free_rows = feat_rows[feat_rows.role == "free"]
        if len(free_rows):
            X_free = preprocessor1.transform(free_rows[feature_cols])
            with torch.no_grad():
                free_probs = torch.sigmoid(model1(torch.tensor(X_free, dtype=torch.float32))).cpu().numpy().flatten()
            best_idx = np.argmax(free_probs)
            free_feats = list(free_rows.iloc[best_idx][feature_cols]) + [free_probs[best_idx]]
        else:
            free_feats = [0] * (len(feature_cols) + 1)

        global_feats = list(part_row[part_level_cols])
        return fixed_feats + load_feats + free_feats + global_feats

    col_names = (
        [f"fixed_{c}" for c in feature_cols] + ["fixed_refine_prob"] +
        [f"load_{c}" for c in feature_cols] + ["load_refine_prob"] +
        [f"free_{c}" for c in feature_cols] + ["free_refine_prob"] +
        part_level_cols
    )
    row = extract_supervised_features()
    df_sup = pd.DataFrame([row], columns=col_names)

    missing_cols = set(preprocessor4.feature_names_in_) - set(df_sup.columns)
    for c in missing_cols:
        df_sup[c] = 0.0
    df_sup = df_sup[preprocessor4.feature_names_in_]

    X4 = preprocessor4.transform(df_sup)
    X4_t = torch.tensor(X4.toarray() if hasattr(X4, "toarray") else X4, dtype=torch.float32)

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

    model4 = GlobalSizeRegressor(X4_t.shape[1])
    state_dict = torch.load("saved_model/model4.pt", map_location="cpu")
    model4.load_state_dict(state_dict)
    model4.eval()

    with torch.no_grad():
        pred4 = model4(X4_t).item()

    print("Model 4 (part-level global_size) prediction:")
    print(f"Predicted global_size_final: {pred4:.4f}")

    # give out the suggestion
    prob_converged = probs2[0,1]
    pred_class = preds2[0]
    global_size = pred4
    face1, face2 = top2_with_local_size.itertuples(index=False)

    if prob_converged > 0.7:
        converge_text = "very likely to converge"
    elif prob_converged > 0.45:
        converge_text = "likely to converge"
    else:
        converge_text = "unlikely to converge"

    sentence = (
        f"The model predicts that the simulation is {converge_text} "
        f"(confidence: {prob_converged:.2%}). "
        f"To improve accuracy, local refinement will be applied on a {face1.ftype.lower()} (ID {face1.face_id}) "
        f"and a {face2.ftype.lower()} (ID {face2.face_id}), "
        f"with predicted mesh sizes {face1.predicted_local_size:.3f} mm and {face2.predicted_local_size:.3f} mm respectively. "
        f"The overall global mesh size is set to {global_size:.3f} mm."
    )
    print(sentence)

    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(global_size))

    gmsh.model.mesh.generate(2)
    node_tags_b, node_coords_b, _ = gmsh.model.mesh.getNodes()
    nodes_b = node_coords_b.reshape(-1, 3)

    types_b, elem_tags_list_b, node_tags_list_b = gmsh.model.mesh.getElements(dim=2)

    elements_b = []
    for etype, ntags in zip(types_b, node_tags_list_b):
        name, dim, order, nnode, *_ = gmsh.model.mesh.getElementProperties(etype)
        if nnode in (3, 4):
            conn = ntags.reshape(-1, nnode).astype(int) - 1
            elements_b.append((etype, conn))

    plot_mesh(
        nodes_b,
        elements_b,
        title="Baseline Mesh (Global Only)",
        highlight_faces=None,
        face_to_nodes=None,
        save_path="output/mesh_global_only.png"
    )

    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", global_size/4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(global_size))

    face_ids    = top2_with_local_size["face_id"].tolist()
    local_sizes = top2_with_local_size["predicted_local_size"].round(3).tolist()

    # Collect curves for distance
    curve_ids = []
    for fid in face_ids:
        surf = fid - 1
        boundary = gmsh.model.getBoundary([(2, surf)], oriented=False)
        for (dim, ctag) in boundary:
            if dim == 1:
                curve_ids.append(ctag)

    # Distance field
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", curve_ids)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

    # Threshold field
    min_local = min(local_sizes)*2
    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", float(min_local))
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", float(global_size))
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", global_size * 2.0)

    # Clamp to prevent refinement **smaller than local size**
    # Min(F_th, global_size) but >= min_local
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th])

    # Set as background
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    node_tags_r, node_coords_r, _ = gmsh.model.mesh.getNodes()
    nodes_r = node_coords_r.reshape(-1, 3)

    types_r, elem_tags_list_r, node_tags_list_r = gmsh.model.mesh.getElements(dim=2)
    elements_r = []
    for etype, ntags in zip(types_r, node_tags_list_r):
        name, dim, order, nnode, *_ = gmsh.model.mesh.getElementProperties(etype)
        if nnode in (3, 4):
            conn = ntags.reshape(-1, nnode).astype(int) - 1
            elements_r.append((etype, conn))

    # Mapping faces→nodes for plotting
    face_to_nodes = {}
    for (dim, sid) in gmsh.model.getEntities(dim=2):
        etypes_s, elem_tags_s, ntags_s = gmsh.model.mesh.getElements(dim=2, tag=sid)
        surface_conn = []
        for etype, ntags in zip(etypes_s, ntags_s):
            _, _, _, nnode, *_ = gmsh.model.mesh.getElementProperties(etype)
            if nnode in (3, 4):
                conn = ntags.reshape(-1, nnode).astype(int) - 1
                surface_conn.extend(conn)
        face_to_nodes[sid] = surface_conn

    highlight_gmsh_ids = [fid - 1 for fid in face_ids]

    # Save refined mesh image
    plot_mesh(
        nodes_r, elements_r,
        title="Refined Mesh (Local + Global)",
        highlight_faces=highlight_gmsh_ids,
        face_to_nodes=face_to_nodes,
        save_path="output/mesh_refined.png"
    )

    gmsh.write("output/mesh_refined.msh")
    gmsh.finalize()

    return sentence


