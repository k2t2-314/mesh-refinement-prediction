from pathlib import Path
import ansys.mechanical.core as mech
from ansys.mechanical.core import App
import json
import numpy as np
import itertools
import csv
from pathlib import Path
import os


part_csv = Path("part_level.csv")
feature_csv = Path("feature_level.csv")


if not part_csv.exists():
    with open(part_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "part_id",
            "num_fillet",
            "num_hole",
            "num_plane",
            "smallest_fillet_r",
            "smallest_hole_r",
            "load_face",
            "load_direction",
            "load_scale",
            "global_size_initial",
            "global_size_final",
            "converged",
            "top2_features",
            "local_refine_needed",
            "local_size"
        ])

if not feature_csv.exists():
    with open(feature_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "part_id",
            "ftype",
            "role",
            "cx", "cy", "cz",
            "nx", "ny", "nz",
            "area",
            "bbox_dx", "bbox_dy", "bbox_dz",
            "dist_load",
            "nlx", "nly", "nlz",
            "dist_fixed",
            "nfx", "nfy", "nfz",
            "global_size_initial",
            "global_size_final",
            "local_size",
            "refine_needed",
            "converged"
        ])

def face_center(face):
    bbox = face.GetBoundingBox()
    return [(bbox[0]+bbox[3])/2, (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]

def match_hole_face(face, patch, tol=0.5, area_tol=15.0):
    c = face_center(face)
    p_c = patch["center"]
    match_count = sum([abs(c[i] - p_c[i]) < tol for i in range(3)])
    area_match = abs(face.Area - patch["area"]) < area_tol
    return match_count == 3 and area_match

def match_fillet_face(face, patch, tol=0.5, area_tol=15.0):
    c = face_center(face)
    p_c = patch["center"]
    match_count = sum(abs(c[i] - p_c[i]) < tol for i in range(3))
    area_match = abs(face.Area - patch["area"]) < area_tol
    return match_count == 3 and area_match

def match_face(face, patch, tol=0.5, area_tol=15.0):
    c = face_center(face)
    p_c = patch["center"]
    match_count = sum(abs(c[i] - p_c[i]) < tol for i in range(3))
    area_match = abs(face.Area - patch["area"]) < area_tol
    return match_count == 3 and area_match


def get_edge_endpoints(edge):
    vs = edge.Vertices
    if len(vs) != 2:
        return None
    p1 = [vs[0].X, vs[0].Y, vs[0].Z]
    p2 = [vs[1].X, vs[1].Y, vs[1].Z]
    return p1, p2

folder = "target_steps"
json_folder = "geo_features"
app = App(globals=globals())

for filename in os.listdir(folder):
    boundary_condition = {
        "direction": [-1,1],
        "load": ["bend_bottom", "tension"],
        "scale": ["low", "medium", "high"]   }

    combos = list(itertools.product(
        boundary_condition["direction"],
        boundary_condition["load"],
        boundary_condition["scale"]
    ))
    for j,c in enumerate(combos):
        filepath = os.path.join(folder, filename)

        raw = os.path.splitext(filename)[0]
        part_id = raw[-3:]
        json_path = os.path.join(json_folder, f"{raw}_features.json")
        with open(json_path) as f:
            features = json.load(f)
        step_path = filepath
        out_folder = "results"

        app.new()
        model = app.Model
        ExtAPI = app.ExtAPI

        analysis = model.AddStaticStructuralAnalysis()


        # 1) import step file
        step_file = Path(step_path)
        gi = model.GeometryImportGroup.AddGeometryImport()
        gi.Import(str(step_file))   # Import geometry into Mechanical

        # 2) boundary setup
        analysis.AnalysisSettings.NumberOfSteps = 1     # Single load step
        sel_mgr = ExtAPI.SelectionManager

        # Assign material to all bodies
        material_assignment = model.Materials.AddMaterialAssignment()
        material_assignment.Material = "Structural Steel"
        mat_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
        mat_sel.Ids = [
            body.GetGeoBody().Id
            for body in model.Geometry.GetChildren(
                Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory.Body, True
            )
        ]
        material_assignment.Location = mat_sel   # Apply material to entire geometry
        
        # Get all faces in model
        geo_data = ExtAPI.DataModel.GeoData
        assemblies = geo_data.Assemblies
        all_faces = []
        for asm in assemblies:
            for part in asm.Parts:
                for body in part.Bodies:
                    if body.Suppressed:
                        continue
                    for face in body.Faces:
                        all_faces.append(face)

        face_to_ids = {}   # FACE_6 → [63]
        face_names = []    # ["FACE_0", "FACE_1", ...]

        for patch in features["faces"]:
            face_name = patch["name"]
            matched_ids = []

            for face in all_faces:
                if match_face(face, patch):
                    matched_ids.append(face.Id)

            if matched_ids:
                face_to_ids[face_name] = matched_ids
                face_names.append(face_name)

                ns = model.AddNamedSelection()
                ns.Name = face_name
                sel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
                sel.Ids = matched_ids
                ns.Location = sel

        mech_id_to_normal = {}
        face_name_to_normal = {
            f["name"]: f["normal"]
            for f in features["faces"]
        }
        for fname, id_list in face_to_ids.items():
            normal = face_name_to_normal[fname]
            for mid in id_list:
                mech_id_to_normal[mid] = normal

        mech_id_to_center = {}

        face_name_to_center = {
            f["name"]: f["center"]
            for f in features["faces"]
        }

        for fname, id_list in face_to_ids.items():
            center = face_name_to_center[fname]
            for mid in id_list:
                mech_id_to_center[mid] = center

        mech_id_to_area = {}

        face_name_to_area = {
            f["name"]: f["area"]
            for f in features["faces"]
        }

        for fname, id_list in face_to_ids.items():
            area = face_name_to_area[fname]
            for mid in id_list:
                mech_id_to_area[mid] = area

        # Step 1: collect hole / fillet face IDs by feature
        hole_faces_by_feature = {}
        for hole in features["circular_holes"]:
            name = hole["name"]
            hole_faces_by_feature[name] = []
            for patch in hole["faces"]:
                for face in all_faces:
                    if match_hole_face(face, patch):
                        hole_faces_by_feature[name].append(face.Id)

        fillet_faces_by_feature = {}
        for fillet in features["fillets"]:
            name = fillet["name"]
            fillet_faces_by_feature[name] = []
            for patch in fillet["faces"]:
                for face in all_faces:
                    if match_fillet_face(face, patch):
                        fillet_faces_by_feature[name].append(face.Id)

        # === Create NS for Holes ===
        hole_ns_names = []

        for hname, fid_list in hole_faces_by_feature.items():
            if not fid_list:
                continue

            ns = model.AddNamedSelection()
            ns.Name = hname
            sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            sel.Ids = fid_list
            ns.Location = sel

            hole_ns_names.append(hname)
            # print(f"[NS] Hole created: {hname} → {fid_list}")

        # === Create NS for Fillets ===
        fillet_ns_names = []

        for fname, fid_list in fillet_faces_by_feature.items():
            if not fid_list:
                continue

            ns = model.AddNamedSelection()
            ns.Name = fname
            sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            sel.Ids = fid_list
            ns.Location = sel

            fillet_ns_names.append(fname)
            # print(f"[NS] Fillet created: {fname} → {fid_list}")

        ns_objects = model.GetChildren(
            Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory.NamedSelection, True
        )

        # Step 2: construct face → feature mapping
        face_feature_map = {}

        # hole faces first
        for hname, flist in hole_faces_by_feature.items():
            for fid in flist:
                face_feature_map[fid] = hname

        # fillet faces
        for fname, flist in fillet_faces_by_feature.items():
            for fid in flist:
                face_feature_map[fid] = fname

        # plane faces (anything not already assigned)
        id_to_face_name = {}
        for fname, id_list in face_to_ids.items():
            for fid in id_list:
                id_to_face_name[fid] = fname

        for face in all_faces:
            fid = face.Id
            if fid not in face_feature_map:
                face_feature_map[fid] = id_to_face_name.get(fid, "PLANE")

        feature_to_ids = {}
        feature_to_ids.update(face_to_ids)
        feature_to_ids.update(hole_faces_by_feature)
        feature_to_ids.update(fillet_faces_by_feature)

        print("face_feature_map =", face_feature_map)


        def unit(v):
            arr = np.array(v, dtype=float)
            n = np.linalg.norm(arr)
            return arr / n if n > 0 else arr

        hole_axis_dir = {}

        for hole in features["circular_holes"]:
            name = hole["name"]
            axis = hole["normal"]
            hole_axis_dir[name] = unit(axis)
        
        # classify faces by dominant axis normal

        plane_faces = [f for f in all_faces if face_feature_map[f.Id].startswith("FACE_")]

        # ===========================
        # STEP 1: filter for straight normals
        # ===========================

        STRAIGHT = 0.97     # threshold

        straight_faces = []
        for f in plane_faces:
            nx, ny, nz = mech_id_to_normal[f.Id]
            if max(abs(nx), abs(ny), abs(nz)) >= STRAIGHT:
                straight_faces.append(f)

        # ===========================
        # STEP 2: dominant axis grouping
        # ===========================

        faces_X = []
        faces_Y = []
        faces_Z = []

        for f in straight_faces:
            nx, ny, nz = mech_id_to_normal[f.Id]
            ax, ay, az = abs(nx), abs(ny), abs(nz)
            if ax >= ay and ax >= az:
                faces_X.append(f)
            elif ay >= ax and ay >= az:
                faces_Y.append(f)
            else:
                faces_Z.append(f)


        # ===========================
        # STEP 3: pick each side
        # ===========================

        # bottom
        bottom_face = min(faces_Z, key=lambda f: mech_id_to_center[f.Id][2])
        bottom_face_ids = [bottom_face.Id]

        # # top
        # top_face = max(faces_Z, key=lambda f: mech_id_to_center[f.Id][2])
        # top_face_ids = [top_face.Id]

        # front
        front_face = min(faces_Y, key=lambda f: mech_id_to_center[f.Id][1])
        front_face_ids = [front_face.Id]

        # back
        back_face = max(faces_Y, key=lambda f: mech_id_to_center[f.Id][1])
        back_face_ids = [back_face.Id]

        # left
        left_face = min(faces_X, key=lambda f: mech_id_to_center[f.Id][0])
        left_face_ids = [left_face.Id]

        # # right
        # right_face = max(faces_X, key=lambda f: mech_id_to_center[f.Id][0])
        # right_face_ids = [right_face.Id]

        print(f"bottom={bottom_face.Id}, left={left_face.Id}, front={front_face.Id}, back={back_face.Id}")

        analysis.Solution.ClearGeneratedData()
        for child in list(analysis.Solution.Children):
            if hasattr(child, "Delete"):
                child.Delete()
        for child in list(analysis.Children):
            if hasattr(child, "Location"):
                child.Delete()

        stress = []
        converged = False
        top2_names = []
        converged_global = False
        top2_feats = None
        stress = []
        bbody = body.GetBoundingBox()
        maxlength = max(bbody[1] - bbody[0], bbody[3] - bbody[2], bbody[5] - bbody[4])
        init_size = max( features["thickness"]*1.25, maxlength / 20 )
        print(f"init_size:{init_size}")
        print(f"maxlength:{maxlength}")
        load_size_multiplier = (features["thickness"] /3 ) * (maxlength / 25)
        print(f"load_size_multiplier:{load_size_multiplier}")
        size = init_size
        local_size = init_size
        direction, load, scale = c
        load_face_center = [0, 0, 0]
        norm_load = []
        fixed_face_center = [0, 0, 0]
        norm_fixed = []
        mesh = model.Mesh
        if load == 'bend_left':
            # Fixed support on the bottom face
            fixed = analysis.AddFixedSupport()
            fix_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            fix_sel.Ids = bottom_face_ids
            fixed.Location = fix_sel

            # Apply force on left face
            force = analysis.AddForce()
            force_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            force_sel.Ids = left_face_ids
            force.Location = force_sel
            force.DefineBy = Ansys.Mechanical.DataModel.Enums.LoadDefineBy.Components
            if scale == "low":
                load_size = load_size_multiplier*50
            elif scale == "medium":
                load_size = load_size_multiplier*150
            elif scale == "high":
                load_size = load_size_multiplier*400
            if direction == -1:
                force.XComponent.Output.DiscreteValues = [Quantity(f"-{load_size} [N]")]
            else:
                force.XComponent.Output.DiscreteValues = [Quantity(f"{load_size} [N]")]

        if load == 'bend_bottom':

            # Fixed support on the left face
            fixed = analysis.AddFixedSupport()
            fix_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            fix_sel.Ids = left_face_ids
            fixed.Location = fix_sel

            # Apply PRESSURE on bottom face
            p = analysis.AddPressure()
            load_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            load_sel.Ids = bottom_face_ids
            p.Location = load_sel

            print(f"Add fix to {left_face_ids}, pressure to {bottom_face_ids}")

            # original scaling
            if scale == "low":
                load_size = load_size_multiplier*100000
            elif scale == "medium":
                load_size = load_size_multiplier*300000
            elif scale == "high":
                load_size = load_size_multiplier*1000000

            # compute pressure
            A = sum(mech_id_to_area[mid] for mid in bottom_face_ids) # m^2
            p_val = load_size / A   # Pa

            if direction == -1:
                p_val = -p_val

            p.Magnitude.Output.DiscreteValues = [Quantity(f"{p_val} [Pa]")]

        if load == 'tension':

            # Fix back face (Ymax)
            fixed = analysis.AddFixedSupport()
            fix_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            fix_sel.Ids = back_face_ids
            fixed.Location = fix_sel

            # Apply PRESSURE on front face (Ymin)
            p = analysis.AddPressure()
            load_sel = sel_mgr.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
            load_sel.Ids = front_face_ids
            p.Location = load_sel

            print(f"Add fix to {back_face_ids}, pressure to {front_face_ids}")

            # original scaling
            if scale == "low":
                load_size = load_size_multiplier*100000
            elif scale == "medium":
                load_size = load_size_multiplier*300000
            elif scale == "high":
                load_size = load_size_multiplier*1000000

            # compute correct area (front face!)
            A = sum(mech_id_to_area[mid] for mid in front_face_ids)

            # convert Force -> Pressure
            p_val = load_size / A   # Pa

            if direction == -1:
                p_val = -p_val

            p.Magnitude.Output.DiscreteValues = [Quantity(f"{p_val} [Pa]")]


        def get_json_face_for_geo_face(geo_face, features):
            for patch in features["faces"]:
                if match_face(geo_face, patch):
                    return patch
            return None

        if load == 'bend_left':
            lf = left_face
            ff = bottom_face
        elif load == 'bend_bottom':
            lf = bottom_face
            ff = left_face
        elif load == 'tension':
            lf = front_face
            ff = back_face

        lf_json = get_json_face_for_geo_face(lf, features)
        ff_json = get_json_face_for_geo_face(ff, features)

        nlx, nly, nlz = lf_json["normal"]
        nfx, nfy, nfz = ff_json["normal"]

        load_face_center = lf_json["center"]
        fixed_face_center = ff_json["center"]


        # add pressure to all circular holes

        for hname in hole_ns_names:
            ns_obj = None
            for ns in ns_objects:
                if ns.Name == hname:
                    ns_obj = ns
                    break

            p_hole = analysis.AddPressure()
            p_hole.Location = ns_obj

            if scale == "low":
                p_hole.Magnitude.Output.DiscreteValues = [Quantity(f"{load_size_multiplier*0.02} [MPa]")]
            elif scale == "medium":
                p_hole.Magnitude.Output.DiscreteValues = [Quantity(f"{load_size_multiplier*0.04} [MPa]")]
            elif scale =="high":
                p_hole.Magnitude.Output.DiscreteValues = [Quantity(f"{load_size_multiplier*0.08} [MPa]")]

        for i in range(5):
            print(f"Running mesh{i} for BC{j} of part{part_id}: direnction = {direction}, load = {load}, scale = {scale}")
            analysis.Solution.ClearGeneratedData()
            analysis.Solution.Children.Clear()

            mesh = model.Mesh
            if i == 0:
                pass
            elif i >= 1 and (
                not top2_feats or
                all(not any(ftype.startswith("FILLET") or ftype.startswith("CIRC_HOLE") 
                            for ftype in ftlist)
                    for ftlist in top2_feats)
            ):
                size *= 0.75
                local_size *= 0.75
                print(f"[GLOBAL refine] global size = {size}")
            elif size >= features["thickness"]:
                size *= 0.8
                local_size *= 0.8
                print(f"[NORMAL refine] global size = {size}")
            mesh.ElementSize = Quantity(f"{size} [mm]")
            if i >= 1 and top2_feats:
                for ns_name, ftypes in zip(top2_names, top2_feats):
                    if any(ft.startswith("FILLET") or ft.startswith("CIRC_HOLE") for ft in ftypes):
                        ns_obj = None
                        for ns in ns_objects:
                            if ns.Name == ns_name:
                                ns_obj = ns
                                break
                        if ns_obj:
                            local_size *= 0.75
                            sizing = mesh.AddSizing()
                            sizing.Type = Ansys.Mechanical.DataModel.Enums.SizingType.ElementSize
                            sizing.ElementSize = Quantity(f"{local_size} [mm]")
                            sizing.Location = ns_obj
                            print(f"[LOCAL refine] {ns_name} ({ftypes}), local={local_size}")
            mesh.GenerateMesh()
            
            # run the simulation

            eqv_stress = analysis.Solution.AddEquivalentStress()

            analysis.Solution.Solve(True)         # Run solver
            analysis.Solution.EvaluateAllResults()  # Evaluate all result objects
            eqv_stress.Activate()
            max_eqv = float(str(eqv_stress.Maximum.Value))
            print("Final max stress:", max_eqv)
            stress.append(max_eqv)

            # # output & visualization
            ExtAPI.Graphics.ExportImage(
                    f"{out_folder}/{part_id}_eqv_stress_{j}_{i}.png"
                )
            # return stress for ML or AMR decisions

            ns_max_map = {}
            for ns in ns_objects:
                if ns.Name in face_names:
                    scoped_stress = analysis.Solution.AddEquivalentStress()
                    scoped_stress.Location = ns
                    analysis.Solution.EvaluateAllResults()

                    max_val = float(str(scoped_stress.Maximum.Value))
                    ns_max_map[ns.Name] = max_val

                    scoped_stress.Delete()

            top2 = sorted(ns_max_map.items(), key=lambda x: x[1], reverse=True)[:2]

            print("Top-2 High Stress Named Selections")
            top2_names = []
            top2_feats = []
            for rank, (ns_name, stress_val) in enumerate(top2, start=1):
                face_ids = face_to_ids.get(ns_name, [])
                feature_types = []
                for fid in face_ids:
                    ftype = face_feature_map.get(fid, "PLANE")
                    feature_types.append(ftype)
                feature_types = list(set(feature_types))
                print(f"Top {rank}: {ns_name}, feature = {feature_types}")
                top2_names.append(ns_name)
                top2_feats.append(feature_types)

            if i >= 2:
                if abs(stress[i] - stress[i-1]) / stress[i-1] < 0.025:
                    converged = True
                    print("converged")
                    break
        
        part_id = part_id
        num_plane  = sum(1 for feat in face_feature_map.values() if feat.startswith("FACE_"))
        num_hole   = sum(1 for feat in face_feature_map.values() if feat.startswith("CIRC_HOLE"))
        num_fillet = sum(1 for feat in face_feature_map.values() if feat.startswith("FILLET"))
        # smallest fillet / hole radius (if exists)
        if features["fillets"]:
            smallest_fillet_r = min(f["radius"] for f in features["fillets"])
        else:
            smallest_fillet_r = 0

        if features["circular_holes"]:
            smallest_hole_r = min(h["radius"] for h in features["circular_holes"])
        else:
            smallest_hole_r = 0
        load_face = load    # "left" or "bottom"
        load_direction = direction
        load_scale = scale
        global_size_initial = init_size
        global_size_final = size
        converged_flag = 1 if converged else 0
        clean_feats = []
        for feats in top2_feats:
            cleaned = []
            for f in feats:
                base = f.rsplit("_", 1)[0]
                cleaned.append(base)
            clean_feats.append(cleaned)
        top2_features = ",".join([",".join(feats) for feats in clean_feats])
        local_refine_needed = 1 if any(ft != ["PLANE"] for ft in top2_feats) else 0
        local_size = local_size

        with open(part_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                    part_id,
                    num_fillet,
                    num_hole,
                    num_plane,
                    smallest_fillet_r,
                    smallest_hole_r,
                    load_face,
                    load_direction,
                    load_scale,
                    global_size_initial,
                    global_size_final,
                    converged_flag,
                    top2_features,
                    local_refine_needed,
                    local_size
                ])
            
        with open(feature_csv, "a", newline="") as f:
            w = csv.writer(f)
            for fid, ftype in face_feature_map.items():
                center = None
                if ftype.startswith("CIRC_HOLE"):
                    for h in features["circular_holes"]:
                        if h["name"] == ftype:
                            center = h["center"]
                            area = h["area"]
                            break
                elif ftype.startswith("FILLET"):
                    for fi in features["fillets"]:
                        if fi["name"] == ftype:
                            center = fi["center"]
                            area = fi["area"]
                            break
                else:
                    aface = next(face for face in all_faces if face.Id == fid)
                    bb = aface.GetBoundingBox()
                    center = [(bb[0] + bb[3]) / 2,
                            (bb[1] + bb[4]) / 2,
                            (bb[2] + bb[5]) / 2]
                    area = aface.Area

                cx, cy, cz = center
                if ftype.startswith("CIRC_HOLE"):
                    feat_json = next(h for h in features["circular_holes"] if h["name"] == ftype)
                elif ftype.startswith("FILLET"):
                    feat_json = next(fi for fi in features["fillets"] if fi["name"] == ftype)
                else:
                    feat_json = next(fa for fa in features["faces"] if fa["name"] == ftype)

                nx, ny, nz = feat_json["normal"]
                
                vals_x = []
                vals_y = []
                vals_z = []

                for face_id in feature_to_ids.get(ftype, []):
                    aface = next(face for face in all_faces if face.Id == face_id)
                    bb = aface.GetBoundingBox()
                    vals_x += [bb[0], bb[3]]
                    vals_y += [bb[1], bb[4]]
                    vals_z += [bb[2], bb[5]]

                bx = max(vals_x) - min(vals_x)
                by = max(vals_y) - min(vals_y)
                bz = max(vals_z) - min(vals_z)

                def dist(a, b):
                    a = np.array(a); b = np.array(b)
                    return float(np.linalg.norm(a - b))

                dist_load  = dist([cx,cy,cz], load_face_center)
                dist_fixed = dist([cx,cy,cz], fixed_face_center)
                refine_needed = 1 if any(ftype in feats for feats in top2_feats) else 0

                if refine_needed:
                    final_local = local_size
                else:
                    final_local = size

                base_ftype = ftype.rsplit("_", 1)[0]
                conv = 1 if converged else 0

                if fid in load_sel.Ids:
                    role = "load"
                elif fid in fix_sel.Ids:
                    role = "fixed"
                else:
                    role = "free"

                w.writerow([
                    part_id,
                    base_ftype,
                    role,
                    cx, cy, cz,
                    nx, ny, nz,
                    area,
                    bx, by, bz,
                    dist_load,
                    # norm_load,
                    nlx, nly, nlz,
                    dist_fixed,
                    # norm_fixed,
                    nfx, nfy, nfz,
                    global_size_initial,
                    global_size_final,
                    local_size,
                    refine_needed,
                    conv
                ])