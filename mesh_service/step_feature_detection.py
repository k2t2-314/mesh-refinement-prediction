import math
import json
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_IN, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Edge
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.BRep import BRep_Tool
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle, GeomAbs_Line
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt, gp_Vec
import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from collections import Counter

def face_bbox_center(face):
    box = Bnd_Box()
    BRepBndLib.Add_s(face, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2
    ]


def load_step_model(step_path):
    """Load a STEP file and return the shape object."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"Failed to load STEP file: {step_path}")
    reader.TransferRoot()
    shape = reader.Shape()
    return shape

def load_step_unit(step_path):
    with open(step_path, "r") as f:
        for line in f:
            if "SI_UNIT" in line:
                print("STEP file unit info:", line.strip())
                return line.strip()

def detect_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0
    while exp.More():
        face_shape = exp.Current()
        face = TopoDS.Face_s(face_shape)
        surf = BRep_Tool.Surface_s(face)
        faces.append({"face": face, "surface": surf})
        idx = idx+1
        exp.Next()
    return faces


def is_hole(face, shape):
    """
    Determine if a cylindrical face is a hole (concave) rather than a boss (convex).
    Basic approach: check if the face normal points toward the solid interior.
    """
    # Get a point and normal on the cylindrical surface
    adaptor = BRepAdaptor_Surface(face, True)
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    u_mid = (u_min + u_max) / 2
    v_mid = (v_min + v_max) / 2
    
    # Get point and normal vector at mid-parameter
    point = adaptor.Value(u_mid, v_mid)
    d1u = gp_Vec()
    d1v = gp_Vec()
    adaptor.D1(u_mid, v_mid, point, d1u, d1v)
    
    # Calculate normal vector (cross product of partial derivatives)
    normal = d1u.Crossed(d1v)
    normal.Normalize()
    
    # Offset point along normal direction
    offset = 0.01  # Small offset distance
    test_point = gp_Pnt(
        point.X() + normal.X() * offset,
        point.Y() + normal.Y() * offset,
        point.Z() + normal.Z() * offset
    )
    
    # Check if the offset point is inside or outside the solid
    classifier = BRepClass3d_SolidClassifier(shape)
    classifier.Perform(test_point, 1e-7)
    
    # If normal points outward, offset point should be outside -> this is a hole
    # If normal points outward, offset point is still inside -> this is a boss
    return classifier.State() == TopAbs_IN


def compute_face_angle_span(face, axis_location, axis_direction):
    axis_loc = np.array(axis_location, dtype=float)
    axis_dir = np.array(axis_direction, dtype=float)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, axis_dir)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    e1 = np.cross(axis_dir, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)
    e2 = e2 / np.linalg.norm(e2)

    angles = []

    edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_exp.More():
        edge_shape = edge_exp.Current()
        vert_exp = TopExp_Explorer(edge_shape, TopAbs_VERTEX)
        while vert_exp.More():
            vert_shape = vert_exp.Current()
            vertex = TopoDS.Vertex_s(vert_shape)
            p = BRep_Tool.Pnt_s(vertex)
            v = np.array([p.X(), p.Y(), p.Z()], dtype=float) - axis_loc

            v_proj = v - np.dot(v, axis_dir) * axis_dir
            norm_v = np.linalg.norm(v_proj)
            if norm_v < 1e-6:
                vert_exp.Next()
                continue

            x = np.dot(v_proj, e1)
            y = np.dot(v_proj, e2)
            theta = math.atan2(y, x)
            angles.append(theta)

            vert_exp.Next()
        edge_exp.Next()

    if not angles:
        return 0.0

    angles = np.array(angles)
    angles = np.mod(angles, 2.0 * math.pi)
    angles.sort()

    diffs = np.diff(angles)
    wrap_gap = (angles[0] + 2.0 * math.pi) - angles[-1]
    max_gap = max(diffs.max(initial=0.0), wrap_gap)

    coverage_rad = 2.0 * math.pi - max_gap
    coverage_deg = math.degrees(coverage_rad)
    return float(coverage_deg)
    

def detect_circular_holes(shape, min_radius=0.5, max_radius=50.0, merge_tolerance=1e-3):
    """
    Detect all cylindrical (round) holes, including through-holes and blind holes.
    Merges faces belonging to the same hole.
    Returns: List of dicts with center, radius, normal, area, etc.
    """
    candidate_faces = []
    idx = 0
    
    for fdict in detect_faces(shape):
        face = fdict["face"]
        surf = fdict["surface"]
        
        # Check if surface type is cylindrical
        surface_type = surf.DynamicType().Name()
        if "Cylindrical" not in surface_type:
            continue
        
        # Use BRepAdaptor_Surface to get cylinder parameters
        adaptor = BRepAdaptor_Surface(face, True)
        
        # Confirm it's a cylinder type
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue
            
        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()
        
        # Only consider reasonable hole sizes
        if not (min_radius <= radius <= max_radius):
            continue
        
        # **Key judgment: check if it's a hole (not a boss)**
        if not is_hole(face, shape):
            continue  # Skip bosses/shafts
        
        # Get axis direction and location
        axis_direction = cylinder.Axis().Direction()
        axis_location = cylinder.Axis().Location()
        axis_dir_vec = [axis_direction.X(), axis_direction.Y(), axis_direction.Z()]
        axis_loc_vec = [axis_location.X(), axis_location.Y(), axis_location.Z()]
        
        # Calculate surface area and center
        gprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gprops)
        area = gprops.Mass()
        center = gprops.CentreOfMass()

        angle_span = compute_face_angle_span(face, axis_loc_vec, axis_dir_vec)
        
        candidate_faces.append({
            "face_id": idx,
            "face": face,
            "radius": float(radius),
            "center": [center.X(), center.Y(), center.Z()],
            "axis_direction": [axis_direction.X(), axis_direction.Y(), axis_direction.Z()],
            "axis_location": [axis_location.X(), axis_location.Y(), axis_location.Z()],
            "area": float(area),
            "angle_span": float(angle_span)
        })

        idx = idx+1
    
    # Group faces that belong to the same hole
    holes = merge_coaxial_cylinders(candidate_faces, merge_tolerance)
    
    print(f"Detected {len(holes)} circular holes (from {len(candidate_faces)} cylindrical faces)")
    return holes


def merge_coaxial_cylinders(faces, tolerance=1e-3, angle_threshold_deg=350.0):
    """
    Merge cylindrical faces that share the same axis (belong to same hole).
    Two cylinders belong to the same hole if:
    1. They have the same radius (within tolerance)
    2. They are coaxial (share the same axis)
    """
    if not faces:
        return []
    
    merged_holes = []
    used = [False] * len(faces)
    
    for i, face1 in enumerate(faces):
        if used[i]:
            continue
        
        # Start a new hole group
        hole_group = [face1]
        used[i] = True
        
        # Find all faces belonging to the same hole
        for j, face2 in enumerate(faces):
            if used[j] or i == j:
                continue
            
            # Check if same radius
            if abs(face1["radius"] - face2["radius"]) > tolerance:
                continue
            
            # Check if coaxial
            if are_coaxial(face1, face2, tolerance):
                hole_group.append(face2)
                used[j] = True
        
        total_angle = sum(f.get("angle_span", 0.0) for f in hole_group)
        total_angle = min(total_angle, 360)
        if total_angle < angle_threshold_deg:
            continue
        
        # Compute merged hole properties
        merged_hole = merge_hole_group(hole_group, i)
        merged_holes.append(merged_hole)
    
    return merged_holes


def are_coaxial(face1, face2, tolerance=1e-3):
    """
    Check if two cylindrical faces are coaxial (share the same axis).
    """
    import numpy as np
    
    # Get axis directions
    dir1 = np.array(face1["axis_direction"])
    dir2 = np.array(face2["axis_direction"])
    
    # Normalize
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)
    
    # Check if parallel (dot product close to ±1)
    dot_product = abs(np.dot(dir1, dir2))
    if abs(dot_product - 1.0) > tolerance:
        return False
    
    # Check if axes are coincident
    # Distance from point on axis1 to axis2
    loc1 = np.array(face1["axis_location"])
    loc2 = np.array(face2["axis_location"])
    
    # Vector from loc1 to loc2
    vec = loc2 - loc1
    
    # Distance from loc2 to axis1 (perpendicular distance)
    distance = np.linalg.norm(vec - np.dot(vec, dir1) * dir1)
    
    return distance < tolerance


def merge_hole_group(hole_group, id):
    """
    Merge multiple faces belonging to the same hole into one hole feature,
    and record all involved faces for downstream mapping.
    """
    import numpy as np

    representative = hole_group[0]
    total_area = sum(face["area"] for face in hole_group)

    centers = np.array([face["center"] for face in hole_group])
    areas = np.array([face["area"] for face in hole_group])
    avg_center = np.average(centers, axis=0, weights=areas)

    # Clean up floating point errors
    avg_center = np.round(avg_center, decimals=6)
    faces_info = [
        {
            "face_id": face.get("face_id"),
            "center": [round(x, 6) for x in face["center"]],
            "area": round(face["area"], 6),
            "angle_span": round(face.get("angle_span", 0.0), 4),
            "normal": [round(x, 6) for x in face.get("normal", [0,0,0])]
        }
        for face in hole_group
    ]

    return {
        "name": f"CIRC_HOLE_{id}",
        "type": "circular_hole",
        "radius": round(representative["radius"], 6),
        "center": avg_center.tolist(),
        "normal": representative["axis_direction"],
        "area": round(total_area, 6),
        "num_faces": len(hole_group),
        "faces": faces_info
    }



def detect_non_circular_holes(shape, min_area=1.0, max_area=1000.0, distance_threshold=10.0):
    """
    Detect non-circular holes (rectangular holes, slots, elongated holes).
    Strategy: Find groups of planar faces that form a closed pocket.
    """
    result = []
    planar_faces = []
    
    # 1. Collect all planar faces that might be part of a hole
    for fdict in detect_faces(shape):
        face = fdict["face"]
        surf = fdict["surface"]
        
        surface_type = surf.DynamicType().Name()
        if "Plane" not in surface_type:
            continue
        
        # Check if this plane is a hole (concave)
        if not is_hole(face, shape):
            continue
        
        # Get face properties
        adaptor = BRepAdaptor_Surface(face, True)
        
        # Confirm it's a plane type
        if adaptor.GetType() != GeomAbs_Plane:
            continue
        
        plane = adaptor.Plane()
        normal = plane.Axis().Direction()
        
        gprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gprops)
        area = gprops.Mass()
        center = gprops.CentreOfMass()
        
        # Filter by area
        if not (min_area <= area <= max_area):
            continue
        
        planar_faces.append({
            "face": face,
            "center": [center.X(), center.Y(), center.Z()],
            "normal": [normal.X(), normal.Y(), normal.Z()],
            "area": float(area)
        })
    
    # 2. Group adjacent planar faces that might form a hole
    holes = group_planar_holes(planar_faces, distance_threshold)
    
    # 3. Classify hole type (rectangular, slot, etc.)
    for hole_group in holes:
        hole_type = classify_planar_hole(hole_group)
        result.append(hole_type)
    
    print(f"Detected {len(result)} non-circular holes (from {len(planar_faces)} planar faces)")
    return result


def group_planar_holes(faces, distance_threshold=10.0):
    """
    Group planar faces that are close to each other and might form a hole.
    """
    if not faces:
        return []
    
    groups = []
    used = [False] * len(faces)
    
    for i, face1 in enumerate(faces):
        if used[i]:
            continue
        
        # Start a new group
        group = [face1]
        used[i] = True
        
        # Find nearby faces with similar normal direction
        for j, face2 in enumerate(faces):
            if used[j] or i == j:
                continue
            
            # Check if normals are similar (parallel faces)
            normal1 = np.array(face1["normal"])
            normal2 = np.array(face2["normal"])
            
            # Normalize
            normal1 = normal1 / np.linalg.norm(normal1)
            normal2 = normal2 / np.linalg.norm(normal2)
            
            dot = abs(np.dot(normal1, normal2))
            
            if dot > 0.9:  # Nearly parallel
                # Check distance between centers
                center1 = np.array(face1["center"])
                center2 = np.array(face2["center"])
                distance = np.linalg.norm(center2 - center1)
                
                if distance < distance_threshold:
                    group.append(face2)
                    used[j] = True
        
        groups.append(group)
    
    return groups


def classify_planar_hole(face_group):
    """
    Classify a group of planar faces into hole types.
    Simple classification based on number of faces and geometry.
    """
    num_faces = len(face_group)
    
    # Calculate total area and average center
    total_area = sum(f["area"] for f in face_group)
    centers = np.array([f["center"] for f in face_group])
    avg_center = np.mean(centers, axis=0)
    
    # Use the first face's normal as representative
    representative_normal = face_group[0]["normal"]
    
    # Simple classification based on number of faces
    if num_faces == 1:
        hole_type = "simple_pocket"
    elif num_faces == 4:
        hole_type = "rectangular_hole"
    elif num_faces == 2:
        hole_type = "slot"
    elif num_faces >= 5:
        hole_type = "complex_pocket"
    else:
        hole_type = "irregular_hole"
    
    return {
        "type": hole_type,
        "center": np.round(avg_center, 6).tolist(),
        "normal": representative_normal,
        "area": round(total_area, 6),
        "num_faces": num_faces
    }

def detect_fillets(shape, min_radius=0.1, max_radius=50.0,
                   classify_type=True, hole_merge_tolerance=1e-3,
                   fillet_merge_tolerance=1e-3,
                   min_total_angle_deg=3.0,
                   max_total_angle_deg=330.0):
    hole_features = detect_circular_holes(
        shape,
        min_radius=min_radius,
        max_radius=max_radius,
        merge_tolerance=hole_merge_tolerance
    )

    hole_cylinders = []
    for h in hole_features:
        hole_cylinders.append({
            "radius": h["radius"],
            "axis_direction": h["normal"],
            "axis_location": h["center"],
        })

    fillet_candidates = []
    face_idx = 0

    for fdict in detect_faces(shape):
        face = fdict["face"]
        surf = fdict["surface"]

        surface_type = surf.DynamicType().Name()
        if "Cylindrical" not in surface_type:
            continue

        adaptor = BRepAdaptor_Surface(face, True)
        if adaptor.GetType() != GeomAbs_Cylinder:
            continue

        cylinder = adaptor.Cylinder()
        radius = cylinder.Radius()

        if not (min_radius <= radius <= max_radius):
            continue

        axis_direction = cylinder.Axis().Direction()
        axis_location = cylinder.Axis().Location()
        axis_dir_vec = [axis_direction.X(), axis_direction.Y(), axis_direction.Z()]
        axis_loc_vec = [axis_location.X(), axis_location.Y(), axis_location.Z()]

        is_on_hole_cylinder = False
        for hc in hole_cylinders:
            if abs(hc["radius"] - radius) > hole_merge_tolerance:
                continue

            if are_coaxial(
                hc,
                {"axis_direction": axis_dir_vec, "axis_location": axis_loc_vec},
                tolerance=hole_merge_tolerance,
            ):
                is_on_hole_cylinder = True
                break

        if is_on_hole_cylinder:
            continue

        gprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gprops)
        area = gprops.Mass()
        center = gprops.CentreOfMass()

        if not is_fillet_surface(face, shape, radius, area,
                                 full_angle_threshold_deg=330.0,
                                 min_angle_threshold_deg=3.0):
            continue

        angle_span = compute_face_angle_span(face, axis_loc_vec, axis_dir_vec)

        if classify_type:
            if is_internal_fillet(face, shape):
                fillet_type = "internal_fillet"
            else:
                fillet_type = "external_fillet"
        else:
            fillet_type = "fillet"

        fillet_candidates.append({
            "face_id": face_idx,
            "face": face,
            "name": f"FILLET_FACE_{face_idx}",
            "type": fillet_type,
            "radius": float(radius),
            "center": [center.X(), center.Y(), center.Z()],
            "axis_direction": axis_dir_vec,
            "axis_location": axis_loc_vec,
            "area": float(area),
            "angle_span": float(angle_span),
        })

        face_idx += 1

    merged_fillets = merge_coaxial_fillets(
        fillet_candidates,
        tolerance=fillet_merge_tolerance,
        min_total_angle_deg=min_total_angle_deg,
        max_total_angle_deg=max_total_angle_deg,
    )

    print(f"Detected {len(merged_fillets)} fillet features (from {len(fillet_candidates)} cylindrical faces)")

    return merged_fillets


def is_fillet_surface(face, shape, radius, area,
                      full_angle_threshold_deg=330.0,
                      min_angle_threshold_deg=3.0):

    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_Cylinder:
        return False

    cylinder = adaptor.Cylinder()
    axis_direction = cylinder.Axis().Direction()
    axis_location = cylinder.Axis().Location()

    axis_dir_vec = [axis_direction.X(), axis_direction.Y(), axis_direction.Z()]
    axis_loc_vec = [axis_location.X(), axis_location.Y(), axis_location.Z()]

    angle_span = compute_face_angle_span(face, axis_loc_vec, axis_dir_vec)

    if angle_span >= full_angle_threshold_deg:
        return False

    if angle_span <= min_angle_threshold_deg:
        return False

    return True


def is_internal_fillet(face, shape):
    """
    Determine if a cylindrical fillet is internal (concave) or external (convex)
    WITHOUT using adaptor.Normal (which does not exist).

    Method:
      - Get point P(u,v) on surface
      - Compute radial direction = (P - axis_location) - projection onto axis_dir
      - Normal = normalized radial direction
      - Offset P slightly along normal, test if inside solid
    """
    adaptor = BRepAdaptor_Surface(face, True)

    # Must be cylinder
    if adaptor.GetType() != GeomAbs_Cylinder:
        return False

    cylinder = adaptor.Cylinder()

    # Cylinder axis
    axis = cylinder.Axis()
    axis_loc = axis.Location()
    axis_dir = axis.Direction()
    axis_dir_vec = gp_Vec(axis_dir.X(), axis_dir.Y(), axis_dir.Z())
    axis_dir_vec.Normalize()

    # Pick midpoint parameter
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    u_mid = 0.5 * (u_min + u_max)
    v_mid = 0.5 * (v_min + v_max)

    # Point on surface
    p = adaptor.Value(u_mid, v_mid)

    # Compute radial vector: r = (p - axis_loc) - projection onto axis_dir
    vec_p_axis = gp_Vec(p.X() - axis_loc.X(),
                        p.Y() - axis_loc.Y(),
                        p.Z() - axis_loc.Z())
    # Projection length
    proj_len = vec_p_axis.Dot(axis_dir_vec)
    # Projection vector
    proj_vec = gp_Vec(axis_dir_vec.X() * proj_len,
                      axis_dir_vec.Y() * proj_len,
                      axis_dir_vec.Z() * proj_len)
    # Radial direction (perpendicular to axis)
    radial_vec = gp_Vec(vec_p_axis.X() - proj_vec.X(),
                        vec_p_axis.Y() - proj_vec.Y(),
                        vec_p_axis.Z() - proj_vec.Z())
    radial_vec.Normalize()

    # Normal = radial direction
    normal = radial_vec

    # Offset point slightly along normal
    offset = 1e-4
    offset_p = gp_Pnt(p.X() + normal.X() * offset,
                      p.Y() + normal.Y() * offset,
                      p.Z() + normal.Z() * offset)

    # Test whether it's inside
    classifier = BRepClass3d_SolidClassifier(shape)
    classifier.Perform(offset_p, 1e-7)

    # If offset point is inside → normal pointing inward → concave fillet
    return classifier.State() == TopAbs_IN


def merge_coaxial_fillets(faces, tolerance=1e-3,
                          min_total_angle_deg=3.0,
                          max_total_angle_deg=330.0):
    if not faces:
        return []

    import numpy as np

    merged_fillets = []
    used = [False] * len(faces)

    for i, f1 in enumerate(faces):
        if used[i]:
            continue

        group = [f1]
        used[i] = True

        for j, f2 in enumerate(faces):
            if used[j] or i == j:
                continue
            if f1.get("type") != f2.get("type"):
                continue

            if abs(f1["radius"] - f2["radius"]) > tolerance:
                continue

            if not are_coaxial(f1, f2, tolerance=tolerance):
                continue

            group.append(f2)
            used[j] = True

        total_angle = sum(g.get("angle_span", 0.0) for g in group)
        total_angle = min(total_angle, 360.0)

        if total_angle < min_total_angle_deg:
            continue

        if total_angle > max_total_angle_deg:
            continue

        merged = merge_fillet_coaxial_group(group, len(merged_fillets), total_angle)
        merged_fillets.append(merged)

    return merged_fillets


def merge_fillet_coaxial_group(fillet_group, group_index, total_angle):
    if not fillet_group:
        return {}

    areas = np.array([f["area"] for f in fillet_group], dtype=float)
    radii = np.array([f["radius"] for f in fillet_group], dtype=float)
    centers = np.array([f["center"] for f in fillet_group], dtype=float)
    axis_dirs = np.array([f["axis_direction"] for f in fillet_group], dtype=float)
    axis_locs = np.array([f["axis_location"] for f in fillet_group], dtype=float)

    total_area = float(areas.sum()) if areas.size > 0 else 0.0

    if total_area > 0:
        avg_center = (centers * areas[:, None]).sum(axis=0) / total_area
        avg_radius = float((radii * areas).sum() / total_area)
        avg_axis_dir = (axis_dirs * areas[:, None]).sum(axis=0) / total_area
        avg_axis_loc = (axis_locs * areas[:, None]).sum(axis=0) / total_area
    else:
        avg_center = centers.mean(axis=0)
        avg_radius = float(radii.mean())
        avg_axis_dir = axis_dirs.mean(axis=0)
        avg_axis_loc = axis_locs.mean(axis=0)

    norm = np.linalg.norm(avg_axis_dir)
    if norm > 0:
        avg_axis_dir = avg_axis_dir / norm

    avg_center = np.round(avg_center, 6)
    avg_axis_dir = np.round(avg_axis_dir, 6)
    avg_axis_loc = np.round(avg_axis_loc, 6)

    faces_info = [
        {
            "face_id": f.get("face_id"),
            "center": [round(x, 6) for x in f["center"]],
            "area": round(f["area"], 6),
            "angle_span": round(f.get("angle_span", 0.0), 4),
            "normal": [round(x, 6) for x in f.get("axis_direction", avg_axis_dir.tolist())],
        }
        for f in fillet_group
    ]

    fillet_type = fillet_group[0].get("type", "fillet")

    return {
        "name": f"FILLET_{group_index}",
        "type": fillet_type,
        "radius": round(avg_radius, 6),
        "center": [float(x) for x in avg_center],
        "normal": [float(x) for x in avg_axis_dir],
        "axis_direction": [float(x) for x in avg_axis_dir],
        "axis_location": [float(x) for x in avg_axis_loc],
        "area": round(total_area, 6),
        "num_faces": len(fillet_group),
        "total_angle": round(float(total_angle), 4),
        "faces": faces_info,
    }



# def classify_fillets_by_size(fillets):
#     """
#     Classify fillets into categories based on radius for FEA mesh refinement.
#     """
#     classification = {
#         "small": [],    # R < 2mm - need very fine mesh
#         "medium": [],   # 2mm <= R < 5mm - need fine mesh
#         "large": []     # R >= 5mm - standard mesh
#     }
    
#     for fillet in fillets:
#         radius = fillet["radius"]
#         if radius < 2.0:
#             classification["small"].append(fillet)
#         elif radius < 5.0:
#             classification["medium"].append(fillet)
#         else:
#             classification["large"].append(fillet)
    
#     print(f"\nFillet Classification:")
#     print(f"  Small fillets (R<2mm): {len(classification['small'])}")
#     print(f"  Medium fillets (2-5mm): {len(classification['medium'])}")
#     print(f"  Large fillets (R>5mm): {len(classification['large'])}")
    
#     return classification

def detect_thickness(shape, sample_density=3, min_thickness=0.01, max_thickness=100.0):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    thickness_results = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    classifier = BRepClass3d_SolidClassifier(shape)
    while exp.More():
        face_shape = exp.Current()
        face = TopoDS.Face_s(face_shape)
        adaptor = BRepAdaptor_Surface(face, True)
        umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
        vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()
        for i in range(sample_density):
            for j in range(sample_density):
                u = umin + (umax - umin) * i / (sample_density - 1)
                v = vmin + (vmax - vmin) * j / (sample_density - 1)
                pnt = adaptor.Value(u, v)

                p = gp_Pnt()
                du = gp_Vec()
                dv = gp_Vec()
                adaptor.D1(u, v, p, du, dv)

                normal_vec = du.Crossed(dv)
                normal_vec.Normalize()
                if normal_vec.Magnitude() < 1e-8:
                    continue
                normal_vec.Normalize()
                thickness = shoot_ray_find_thickness(
                    shape, pnt, normal_vec, min_thickness, max_thickness
                )
                if thickness:
                    thickness_results.append({
                        "point": [pnt.X(), pnt.Y(), pnt.Z()],
                        "normal": [normal_vec.X(), normal_vec.Y(), normal_vec.Z()],
                        "thickness": thickness
                    })
        exp.Next()
    return thickness_results

def shoot_ray_find_thickness(shape, point, normal, min_thickness, max_thickness):
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier

    step = min_thickness / 2
    max_dist = max_thickness
    for dist in np.arange(step, max_dist, step):
        test_point = gp_Pnt(
            point.X() + normal.X() * dist,
            point.Y() + normal.Y() * dist,
            point.Z() + normal.Z() * dist,
        )
        classifier = BRepClass3d_SolidClassifier(shape)
        classifier.Perform(test_point, 1e-7)
        if classifier.State() != TopAbs_IN:
            pos_thick = dist
            break
    else:
        pos_thick = max_dist
    for dist in np.arange(step, max_dist, step):
        test_point = gp_Pnt(
            point.X() - normal.X() * dist,
            point.Y() - normal.Y() * dist,
            point.Z() - normal.Z() * dist,
        )
        classifier = BRepClass3d_SolidClassifier(shape)
        classifier.Perform(test_point, 1e-7)
        if classifier.State() != TopAbs_IN:
            neg_thick = dist
            break
    else:
        neg_thick = max_dist
    return pos_thick + neg_thick

def mode_thickness(shape, decimal=2, threshold=0.1):
    thickness_results = detect_thickness(shape, sample_density=3, min_thickness=0.01, max_thickness=100.0)
    vals = [round(float(r['thickness']), decimal) for r in thickness_results if float(r['thickness']) >= threshold]
    counter = Counter(vals)
    mode_val, count = counter.most_common(1)[0]
    return mode_val

def detect_all_edges(shape):
    """
    Detect all UNIQUE edges shared by two faces.
    Stores: name, start point, end point, angle, normals, type.
    """

    # 1. Collect all faces
    faces = []
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        faces.append(TopoDS.Face_s(face_exp.Current()))
        face_exp.Next()

    results = []
    seen = set()      # edge deduplication via TShape()
    idx = 0           # MUST initialize once

    # 2. Iterate edges
    edge_exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_exp.More():
        edge = TopoDS.Edge_s(edge_exp.Current())

        # --- dedup by underlying TShape ---
        tid = hash(edge.TShape())
        if tid in seen:
            edge_exp.Next()
            continue
        seen.add(tid)

        # --- find adjacent faces ---
        adjacent_faces = []
        for face in faces:
            ex2 = TopExp_Explorer(face, TopAbs_EDGE)
            while ex2.More():
                if ex2.Current().IsSame(edge):
                    adjacent_faces.append(face)
                    break
                ex2.Next()

        if len(adjacent_faces) != 2:
            edge_exp.Next()
            continue

        # --- compute normals & dihedral angle ---
        normals = []
        for face in adjacent_faces:
            adaptor = BRepAdaptor_Surface(face, True)
            u = 0.5 * (adaptor.FirstUParameter() + adaptor.LastUParameter())
            v = 0.5 * (adaptor.FirstVParameter() + adaptor.LastVParameter())
            du = gp_Vec()
            dv = gp_Vec()
            temp_p = gp_Pnt()
            adaptor.D1(u, v, temp_p, du, dv)
            n = du.Crossed(dv)
            if n.Magnitude() > 1e-6:
                n.Normalize()
                normals.append(n)
            else:
                normals.append(None)

        if None in normals:
            edge_exp.Next()
            continue

        # angle
        import numpy as np
        dot = np.clip(normals[0].Dot(normals[1]), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(dot)))

        # skip perfectly smooth edges
        if abs(angle_deg - 180.0) < 1e-3:
            edge_exp.Next()
            continue

        # --- get start / end points ---
        verts = []
        vexp = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vexp.More():
            v = TopoDS.Vertex_s(vexp.Current())
            p = BRep_Tool.Pnt_s(v)
            verts.append([p.X(), p.Y(), p.Z()])
            vexp.Next()

        if len(verts) < 2:
            edge_exp.Next()
            continue

        start = verts[0]
        end   = verts[-1]

        # --- classify edge type ---
        curve_adaptor = BRepAdaptor_Curve(edge)
        from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle
        t = curve_adaptor.GetType()

        if t == GeomAbs_Line:
            e_type = "line"
        elif t == GeomAbs_Circle:
            e_type = "circle"
        else:
            e_type = "other"

        # --- store result ---
        results.append({
            "name": f"EDGE_{idx}",
            "start": start,
            "end": end,
            "angle": angle_deg,
            "normals": [
                [normals[0].X(), normals[0].Y(), normals[0].Z()],
                [normals[1].X(), normals[1].Y(), normals[1].Z()]
            ],
            "edge_type": e_type
        })

        idx += 1      # safe!

        edge_exp.Next()

    return results




def detect_all_faces(shape):
    faces = []
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0

    while face_exp.More():
        face_shape = face_exp.Current()
        face = TopoDS.Face_s(face_shape)

        adaptor = BRepAdaptor_Surface(face, True)

        # sample param
        umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
        vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()
        u = (umin + umax) * 0.5
        v = (vmin + vmax) * 0.5

        # center
        

        # normal
        du = gp_Vec()
        dv = gp_Vec()
        temp_p = gp_Pnt()
        adaptor.D1(u, v, temp_p, du, dv)
        normal = du.Crossed(dv)
        if normal.Magnitude() > 1e-8:
            normal.Normalize()

        # --- NEW: compute area ---
        gprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, gprops)
        area = gprops.Mass()
        center = face_bbox_center(face)

        faces.append({
            "name": f"FACE_{face_id}",
            "face_id": face_id,
            "center": center,
            "normal": [normal.X(), normal.Y(), normal.Z()],
            "area": float(area)     # ← added
        })

        face_id += 1
        face_exp.Next()

    return faces


