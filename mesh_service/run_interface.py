## MVP 1 FILE!
from .step_feature_detection import (
    load_step_model,
    load_step_unit,
    detect_faces,
    detect_circular_holes,
    detect_non_circular_holes,
    detect_fillets,
    mode_thickness,
    detect_all_edges,
    detect_all_faces,
)
import os, json
import pandas as pd
import gmsh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
import joblib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Matplotlib static PNG plotter (unchanged)
# ---------------------------------------------------------------------
def plot_mesh(
    nodes,
    elements,
    title,
    highlight_faces=None,
    face_to_nodes=None,
    save_path="mesh.png",
):
    """
    Static 3D line-plot of the surface mesh for PNG export.
    nodes: (N,3)
    elements: list[(etype, conn)], conn is array of 0-based node indices
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=45)

    # normal mesh (black)
    for (etype, conn) in elements:
        for e in conn:
            pts = nodes[e][:, :3]
            pts = np.vstack([pts, pts[0]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "k-", linewidth=0.4)

    # optional red overlay for selected faces
    if highlight_faces and face_to_nodes:
        for sf in highlight_faces:
            if sf not in face_to_nodes:
                continue
            face_nodes = face_to_nodes[sf]
            for fn in face_nodes:
                pts = nodes[fn][:, :3]
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", linewidth=1.2)

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Face list helper for GUI (unchanged)
# ---------------------------------------------------------------------
def extract_faces_for_gui(step_path):
    """
    Load the STEP file, detect faces, and return a list of
    label/metadata dicts that the GUI can use.
    """
    shape = load_step_model(step_path)
    faces = detect_all_faces(shape)

    face_list = []
    for f in faces:
        center = f["center"]
        normal = f["normal"]
        area = f.get("area", 0.0)
        label = f"Face {f['face_id']} | normal={np.round(normal,3)} | area={area:.2f}"

        face_list.append(
            {
                "label": label,
                "face_id": f["face_id"],
                "center": list(center),
                "normal": list(normal),
                "area": float(area),
            }
        )

    return face_list


# ---------------------------------------------------------------------
# Helpers for Plotly 3D mesh
# ---------------------------------------------------------------------
def mesh_edges_from_triangles(triangles):
    """
    Given (M,3) triangle connectivity, return unique edges as (E,2).
    """
    edges = set()
    for tri in triangles:
        a, b, c = tri
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.array(list(edges), dtype=int)


def make_plotly_mesh(
    nodes,
    triangles,
    title="Refined Mesh",
    view_mode="wireframe",
    refined_tri_mask=None,
    vertex_refinement_values=None,
):
    """
    Build an interactive Plotly figure.

    Parameters
    ----------
    nodes : (N,3) ndarray
    triangles : (M,3) ndarray of 0-based node indices
    view_mode : {"wireframe", "highlight", "heatmap"}
        - "wireframe": light-blue surface + black edges
        - "highlight": base surface + red overlay on refined triangles
        - "heatmap": surface colored by vertex_refinement_values
    refined_tri_mask : (M,) bool array
        True for triangles in locally-refined region (used in "highlight").
    vertex_refinement_values : (N,) array
        Per-node refinement intensity used for "heatmap".
        Higher values → more refined → hotter colors.
    """
    nodes = np.asarray(nodes)
    triangles = np.asarray(triangles, dtype=int)
    edges = mesh_edges_from_triangles(triangles)
    n_nodes = nodes.shape[0]

    if refined_tri_mask is None:
        refined_tri_mask = np.zeros(triangles.shape[0], dtype=bool)

    if vertex_refinement_values is None:
        vertex_refinement_values = np.zeros(n_nodes, dtype=float)

    fig = go.Figure()

    # ----------------------------------------
    # Base surface for all modes
    # ----------------------------------------
    if view_mode == "heatmap":
        # Heatmap using per-vertex refinement values
        fig.add_trace(
            go.Mesh3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                intensity=vertex_refinement_values,
                colorscale="Turbo",
                showscale=True,
                flatshading=True,
                opacity=0.9,
                showlegend=False,
                name="Local mesh size heatmap",
            )
        )
    else:
        # Simple uniform surface color
        fig.add_trace(
            go.Mesh3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color="lightblue",
                flatshading=True,
                lighting=dict(
                    ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.9
                ),
                lightposition=dict(x=0, y=0, z=1),
                opacity=0.45 if view_mode == "wireframe" else 0.55,
                showlegend=False,
                showscale=False,
                name="Surface",
            )
        )

    # ----------------------------------------
    # Highlight overlay (local refinement in red)
    # ----------------------------------------
    if view_mode == "highlight" and refined_tri_mask.any():
        ref_tris = triangles[refined_tri_mask]
        fig.add_trace(
            go.Mesh3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                i=ref_tris[:, 0],
                j=ref_tris[:, 1],
                k=ref_tris[:, 2],
                color="red",
                opacity=0.9,
                flatshading=True,
                showscale=False,
                name="Locally refined",
            )
        )

    # ----------------------------------------
    # Wireframe edges for all modes
    # ----------------------------------------
    edge_width = 2 if view_mode != "heatmap" else 1
    edge_color = "black" if view_mode != "heatmap" else "rgba(0,0,0,0.4)"

    for e0, e1 in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[nodes[e0, 0], nodes[e1, 0]],
                y=[nodes[e0, 1], nodes[e1, 1]],
                z=[nodes[e0, 2], nodes[e1, 2]],
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def get_result(
    step_path,
    thickness,
    load_face,
    load_direction,
    load_scale,
    selected_face_label=None,
    faces_json=None,
    view_mode="wireframe",  # NEW optional flag, default keeps current behavior
):
    # -----------------------------------------------------------------
    # STEP → JSON features
    # -----------------------------------------------------------------
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
            "faces": faces,
        }
        out_path = "output/example.json"
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)

    step_to_json(step_path, thickness)

    # -----------------------------------------------------------------
    # Load geometry into gmsh
    # -----------------------------------------------------------------
    gmsh.open(step_path)
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surface_info = []
    for dim, tag in surfaces:
        mass_props = gmsh.model.occ.getMass(dim, tag)
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        surface_info.append({"tag": tag, "center": com, "area": mass_props})

    with open("output/example.json") as f:
        features = json.load(f)
    faces = features["faces"]

    num_fillet = len(features["fillets"])
    num_hole = len(features["circular_holes"])
    num_faces = len(features["faces"])
    num_plane = num_faces - num_fillet - num_hole

    smallest_fillet_r = min(
        (f["radius"] for f in features["fillets"]), default=0.0
    )
    smallest_hole_r = min(
        (h["radius"] for h in features["circular_holes"]), default=0.0
    )

    print(
        f"num_fillet: {num_fillet}, num_hole: {num_hole}, num_plane: {num_plane}"
    )
    print(
        f"smallest_fillet_r: {smallest_fillet_r}, smallest_hole_r: {smallest_hole_r}"
    )

    # -----------------------------------------------------------------
    # Bounding box → max dimension
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Heuristic face roles (bottom/left/front/back)
    # -----------------------------------------------------------------
    def find_the_faces(step_path):
        shape = load_step_model(step_path)
        faces_local = detect_all_faces(shape)
        z_vals = [f["center"][2] for f in faces_local]
        x_vals = [f["center"][0] for f in faces_local]
        y_vals = [f["center"][1] for f in faces_local]

        idx_bottom = z_vals.index(min(z_vals))
        idx_left = x_vals.index(min(x_vals))
        idx_front = y_vals.index(min(y_vals))
        idx_back = y_vals.index(max(y_vals))

        return {
            "bottom": faces_local[idx_bottom],
            "left": faces_local[idx_left],
            "front": faces_local[idx_front],
            "back": faces_local[idx_back],
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

    # classify face types
    face_type_dict = {}
    for hole in features["circular_holes"]:
        for f in hole["faces"]:
            face_type_dict[f["face_id"]] = "CIRC_HOLE"
    for fillet in features["fillets"]:
        for f in fillet["faces"]:
            face_type_dict[f["face_id"]] = "FILLET"
    for f in features["faces"]:
        if f["face_id"] not in face_type_dict:
            face_type_dict[f["face_id"]] = "FACE"

    # -----------------------------------------------------------------
    # Choose load/fixed faces (GUI-aware)
    # -----------------------------------------------------------------
    load_face_data = None
    fix_face_data = None

    if selected_face_label is not None and faces_json is not None and faces_json != "":
        faces_from_gui = json.loads(faces_json)
        label_to_face = {f["label"]: f for f in faces_from_gui}

        if selected_face_label not in label_to_face:
            raise ValueError(
                f"Selected load face '{selected_face_label}' not found in faces_json."
            )

        load_face_data = label_to_face[selected_face_label]

        load_center = np.array(load_face_data["center"], dtype=float)
        all_faces_gui = list(label_to_face.values())
        distances = [
            np.linalg.norm(np.array(f["center"], dtype=float) - load_center)
            for f in all_faces_gui
        ]
        fix_face_data = all_faces_gui[int(np.argmax(distances))]
    else:
        if load_face == "bend_bottom":
            load_face_data = face_dict["bottom"]
            fix_face_data = face_dict["left"]
        elif load_face == "tension":
            load_face_data = face_dict["back"]
            fix_face_data = face_dict["front"]
        else:
            raise ValueError(
                f"No GUI selection and unsupported load_face value '{load_face}'."
            )

    load_face_center = np.array(load_face_data["center"])
    load_face_normal = np.array(load_face_data["normal"])
    fix_face_center = np.array(fix_face_data["center"])
    fix_face_normal = np.array(fix_face_data["normal"])

    # -----------------------------------------------------------------
    # BBox per CAD face
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Build per-face feature table
    # -----------------------------------------------------------------
    rows1 = []
    for f in faces:
        row = {}
        row["face_id"] = f["face_id"]
        row["ftype"] = face_type_dict[f["face_id"]]
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

    df_feature = df_feature.fillna(0.0)
    df_part = df_part.fillna(0.0)

    # -----------------------------------------------------------------
    # Model 1: refine_needed classifier
    # -----------------------------------------------------------------
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
                nn.ReLU(),
            )
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.encoder(x)
            return self.head(h)

    preprocessor1 = joblib.load("saved_model/preprocessor1.pkl")
    feature_cols1 = [
        c
        for c in df_feature.columns
        if c
        not in [
            "part_id",
            "refine_needed",
            "converged",
            "local_size",
            "global_size_initial",
            "global_size_final",
            "load_id",
        ]
    ]
    X1 = preprocessor1.transform(df_feature[feature_cols1])
    X1_t = torch.tensor(
        X1.toarray() if hasattr(X1, "toarray") else X1, dtype=torch.float32
    )

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

    # -----------------------------------------------------------------
    # Model 2: part-level convergence
    # -----------------------------------------------------------------
    preprocessor2 = joblib.load("saved_model/preprocessor2.pkl")

    feature_cols = [
        col
        for col in df_feature.columns
        if col
        not in [
            "part_id",
            "load_id",
            "refine_needed",
            "converged",
            "local_size",
            "global_size_initial",
            "global_size_final",
        ]
    ]
    part_level_cols = [
        col
        for col in df_part.columns
        if col
        not in [
            "part_id",
            "converged",
            "load_id",
            "top2_features",
            "global_size_initial",
            "global_size_final",
            "local_refine_needed",
            "local_size",
        ]
    ]

    def extract_supervised_features_model2():
        part_row_local = df_part.iloc[0]
        feat_rows = df_feature

        # fixed
        fixed_row = feat_rows[feat_rows.role == "fixed"]
        if len(fixed_row):
            X_fixed = preprocessor1.transform(fixed_row[feature_cols])
            with torch.no_grad():
                fixed_prob = torch.sigmoid(
                    model1(torch.tensor(X_fixed, dtype=torch.float32))
                ).item()
            fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
        else:
            fixed_feats = [0] * (len(feature_cols) + 1)

        # load
        load_row = feat_rows[feat_rows.role == "load"]
        if len(load_row):
            X_load = preprocessor1.transform(load_row[feature_cols])
            with torch.no_grad():
                load_prob = torch.sigmoid(
                    model1(torch.tensor(X_load, dtype=torch.float32))
                ).item()
            load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
        else:
            load_feats = [0] * (len(feature_cols) + 1)

        # free
        free_rows = feat_rows[feat_rows.role == "free"]
        if len(free_rows):
            X_free = preprocessor1.transform(free_rows[feature_cols])
            with torch.no_grad():
                free_probs = torch.sigmoid(
                    model1(torch.tensor(X_free, dtype=torch.float32))
                ).cpu().numpy().flatten()
            best_idx = np.argmax(free_probs)
            free_feats = (
                list(free_rows.iloc[best_idx][feature_cols])
                + [free_probs[best_idx]]
            )
        else:
            free_feats = [0] * (len(feature_cols) + 1)

        global_feats = list(part_row_local[part_level_cols])
        return fixed_feats + load_feats + free_feats + global_feats

    col_names = (
        [f"fixed_{c}" for c in feature_cols]
        + ["fixed_refine_prob"]
        + [f"load_{c}" for c in feature_cols]
        + ["load_refine_prob"]
        + [f"free_{c}" for c in feature_cols]
        + ["free_refine_prob"]
        + part_level_cols
    )

    row = extract_supervised_features_model2()
    df_sup = pd.DataFrame([row], columns=col_names)

    missing_cols = set(preprocessor2.feature_names_in_) - set(df_sup.columns)
    for c in missing_cols:
        df_sup[c] = 0.0
    df_sup = df_sup[preprocessor2.feature_names_in_]

    X2 = preprocessor2.transform(df_sup)
    print("X2 contains NaN?", np.isnan(X2).any())
    X2_t = torch.tensor(
        X2.toarray() if hasattr(X2, "toarray") else X2, dtype=torch.float32
    )

    class ConvergeClassifier(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
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
    print("Probability not converged:", probs2[:, 0])
    print("Probability converged:", probs2[:, 1])
    print("Predicted class (0: unconverged, 1: converged):", preds2)

    # -----------------------------------------------------------------
    # Model 3: local size regressor
    # -----------------------------------------------------------------
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
                nn.ReLU(),
            )
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.encoder(x)
            return self.head(h)

    preprocessor3 = joblib.load("saved_model/preprocessor3.pkl")
    feature_cols3 = [
        c
        for c in df_feature.columns
        if c
        not in [
            "part_id",
            "refine_needed",
            "converged",
            "local_size",
            "global_size_initial",
            "global_size_final",
            "load_id",
        ]
    ]
    X3 = preprocessor3.transform(df_feature[feature_cols3])
    X3_t = torch.tensor(
        X3.toarray() if hasattr(X3, "toarray") else X3, dtype=torch.float32
    )

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
    top2_with_local_size["predicted_local_size"] = df_feature.loc[
        top2.index, "predicted_local_size"
    ]

    print("Top-2 faces needing refinement and their predicted local sizes:")
    print(
        top2_with_local_size[
            ["face_id", "ftype", "predicted_local_size"]
        ].to_string(index=False)
    )
    df_feature.drop(columns=["refine_prob", "predicted_local_size"], inplace=True)

    # -----------------------------------------------------------------
    # Model 4: global size regressor
    # -----------------------------------------------------------------
    preprocessor4 = joblib.load("saved_model/preprocessor4.pkl")

    feature_cols = [
        col
        for col in df_feature.columns
        if col
        not in [
            "part_id",
            "load_id",
            "refine_needed",
            "converged",
            "local_size",
            "global_size_initial",
            "global_size_final",
        ]
    ]
    part_level_cols = [
        col
        for col in df_part.columns
        if col
        not in [
            "part_id",
            "converged",
            "load_id",
            "top2_features",
            "global_size_initial",
            "global_size_final",
            "local_refine_needed",
            "local_size",
        ]
    ]

    def extract_supervised_features_model4():
        part_row_local = df_part.iloc[0]
        feat_rows = df_feature

        fixed_row = feat_rows[feat_rows.role == "fixed"]
        if len(fixed_row):
            X_fixed = preprocessor1.transform(fixed_row[feature_cols])
            with torch.no_grad():
                fixed_prob = torch.sigmoid(
                    model1(torch.tensor(X_fixed, dtype=torch.float32))
                ).item()
            fixed_feats = list(fixed_row.iloc[0][feature_cols]) + [fixed_prob]
        else:
            fixed_feats = [0] * (len(feature_cols) + 1)

        load_row = feat_rows[feat_rows.role == "load"]
        if len(load_row):
            X_load = preprocessor1.transform(load_row[feature_cols])
            with torch.no_grad():
                load_prob = torch.sigmoid(
                    model1(torch.tensor(X_load, dtype=torch.float32))
                ).item()
            load_feats = list(load_row.iloc[0][feature_cols]) + [load_prob]
        else:
            load_feats = [0] * (len(feature_cols) + 1)

        free_rows = feat_rows[feat_rows.role == "free"]
        if len(free_rows):
            X_free = preprocessor1.transform(free_rows[feature_cols])
            with torch.no_grad():
                free_probs = torch.sigmoid(
                    model1(torch.tensor(X_free, dtype=torch.float32))
                ).cpu().numpy().flatten()
            best_idx = np.argmax(free_probs)
            free_feats = (
                list(free_rows.iloc[best_idx][feature_cols])
                + [free_probs[best_idx]]
            )
        else:
            free_feats = [0] * (len(feature_cols) + 1)

        global_feats = list(part_row_local[part_level_cols])
        return fixed_feats + load_feats + free_feats + global_feats

    col_names = (
        [f"fixed_{c}" for c in feature_cols]
        + ["fixed_refine_prob"]
        + [f"load_{c}" for c in feature_cols]
        + ["load_refine_prob"]
        + [f"free_{c}" for c in feature_cols]
        + ["free_refine_prob"]
        + part_level_cols
    )
    row = extract_supervised_features_model4()
    df_sup = pd.DataFrame([row], columns=col_names)

    missing_cols = set(preprocessor4.feature_names_in_) - set(df_sup.columns)
    for c in missing_cols:
        df_sup[c] = 0.0
    df_sup = df_sup[preprocessor4.feature_names_in_]

    X4 = preprocessor4.transform(df_sup)
    X4_t = torch.tensor(
        X4.toarray() if hasattr(X4, "toarray") else X4, dtype=torch.float32
    )

    class GlobalSizeRegressor(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
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

    # -----------------------------------------------------------------
    # Human-readable sentence
    # -----------------------------------------------------------------
    prob_converged = probs2[0, 1]
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

    # -----------------------------------------------------------------
    # Baseline global-only mesh (for PNG)
    # -----------------------------------------------------------------
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(global_size))

    gmsh.model.mesh.generate(2)
    node_tags_b, node_coords_b, _ = gmsh.model.mesh.getNodes()
    nodes_b = node_coords_b.reshape(-1, 3)

    types_b, elem_tags_list_b, node_tags_list_b = gmsh.model.mesh.getElements(
        dim=2
    )

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
        save_path="output/mesh_global_only.png",
    )

    # -----------------------------------------------------------------
    # Refined mesh with background field
    # -----------------------------------------------------------------
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", global_size / 4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(global_size))

    face_ids = top2_with_local_size["face_id"].tolist()
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
    min_local = min(local_sizes) * 2
    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", float(min_local))
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", float(global_size))
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", global_size * 2.0)

    # Min field (clamped)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th])

    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.occ.synchronize()

    # -----------------------------------------------------------------
    # Extract FE mesh nodes & triangles (compact indexing)
    # -----------------------------------------------------------------
    all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    all_nodes = all_node_coords.reshape(-1, 3)

    types_all, elem_tags_all, elem_nodes_all = gmsh.model.mesh.getElements()

    used_node_tags = np.unique(
        np.concatenate([arr for arr in elem_nodes_all])
    )
    used_idx = used_node_tags - 1
    nodes_r = all_nodes[used_idx]

    tag_to_new_index = {tag: i for i, tag in enumerate(used_node_tags)}

    triangle_list = []
    for etype, conn in zip(types_all, elem_nodes_all):
        name, dim, order, nnode, *_ = gmsh.model.mesh.getElementProperties(etype)
        if dim != 2:
            continue

        if nnode == 3:
            raw = conn.reshape(-1, 3)
            remapped = np.vectorize(tag_to_new_index.get)(raw)
            triangle_list.extend(remapped)
        elif nnode == 4:
            raw = conn.reshape(-1, 4)
            remapped = np.vectorize(tag_to_new_index.get)(raw)
            for quad in remapped:
                triangle_list.append([quad[0], quad[1], quad[2]])
                triangle_list.append([quad[0], quad[2], quad[3]])

    triangle_array = np.array(triangle_list, dtype=int)

    print("Triangles extracted:", triangle_array.shape)
    print("FE MESH NODES:", len(nodes_r))
    print("FE MESH TRIANGLES:", len(triangle_array))

    # For PNG plotter
    elements_r = [(0, triangle_array)]

    # -----------------------------------------------------------------
    # Mapping gmsh surfaces → local node indices (for PNG highlight)
    # -----------------------------------------------------------------
    face_to_nodes = {}
    for (dim, sid) in gmsh.model.getEntities(dim=2):
        etypes_s, elem_tags_s, ntags_s = gmsh.model.mesh.getElements(
            dim=2, tag=sid
        )
        surface_conn_local = []
        for etype, ntags in zip(etypes_s, ntags_s):
            _, _, _, nnode, *_ = gmsh.model.mesh.getElementProperties(etype)
            if nnode in (3, 4):
                raw = ntags.reshape(-1, nnode)
                # map gmsh node tags → compact indices
                mapped = np.vectorize(tag_to_new_index.get)(raw)
                surface_conn_local.extend(mapped)
        face_to_nodes[sid] = surface_conn_local

    highlight_gmsh_ids = [fid - 1 for fid in face_ids]

    # -----------------------------------------------------------------
    # Local refinement mask (which triangles are "near" refined faces)
    # -----------------------------------------------------------------
    refined_nodes_set = set()
    for sid in highlight_gmsh_ids:
        if sid in face_to_nodes:
            for conn in face_to_nodes[sid]:
                for n in conn:
                    refined_nodes_set.add(int(n))

    refined_tri_mask = np.zeros(triangle_array.shape[0], dtype=bool)
    for t_idx, tri in enumerate(triangle_array):
        if any(int(v) in refined_nodes_set for v in tri):
            refined_tri_mask[t_idx] = True

    # -----------------------------------------------------------------
    # Heatmap: per-vertex "local mesh size" (smaller → more refined)
    # -----------------------------------------------------------------
    n_nodes = nodes_r.shape[0]
    n_tris = triangle_array.shape[0]

    tri_sizes = np.zeros(n_tris, dtype=float)
    for t_idx, (a, b, c) in enumerate(triangle_array):
        pa, pb, pc = nodes_r[[a, b, c]]
        area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))
        # characteristic length ~ sqrt(area)
        tri_sizes[t_idx] = np.sqrt(area + 1e-16)

    vertex_sum = np.zeros(n_nodes, dtype=float)
    vertex_count = np.zeros(n_nodes, dtype=float)
    for t_idx, tri in enumerate(triangle_array):
        s = tri_sizes[t_idx]
        for v in tri:
            vertex_sum[v] += s
            vertex_count[v] += 1.0

    vertex_size = vertex_sum / np.maximum(vertex_count, 1.0)

    smin, smax = float(vertex_size.min()), float(vertex_size.max())
    if smax > smin:
        # invert so small elements → high intensity
        vertex_refinement_values = 1.0 - (vertex_size - smin) / (smax - smin)
    else:
        vertex_refinement_values = np.zeros_like(vertex_size)

    # -----------------------------------------------------------------
    # Save refined mesh PNG with red outlines on refined faces
    # -----------------------------------------------------------------
    plot_mesh(
        nodes_r,
        elements_r,
        title="Refined Mesh (Local + Global)",
        highlight_faces=highlight_gmsh_ids,
        face_to_nodes=face_to_nodes,
        save_path="output/mesh_refined.png",
    )

    gmsh.write("output/mesh_refined.msh")

    # -----------------------------------------------------------------
    # Interactive Plotly figure (wireframe / highlight / heatmap)
    # -----------------------------------------------------------------
    interactive_fig = make_plotly_mesh(
        nodes_r,
        triangle_array,
        title="Refined Mesh (Local + Global)",
        view_mode=view_mode,
        refined_tri_mask=refined_tri_mask,
        vertex_refinement_values=vertex_refinement_values,
    )

    msh_path = "output/mesh_refined.msh"
    return (
        sentence,
        "output/mesh_global_only.png",
        "output/mesh_refined.png",
        interactive_fig,
        msh_path,
    )
