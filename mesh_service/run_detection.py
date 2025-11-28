from step_feature_detection import load_step_model, load_step_unit, detect_faces, detect_circular_holes, detect_non_circular_holes, detect_fillets, mode_thickness, detect_all_edges, detect_all_faces
import os, json

def detect_and_export(input_folder_path, export_path):
    os.makedirs(export_path, exist_ok=True)
    step_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.step', '.stp'))]
    if not step_files:
        print("No STEP files found in", input_folder_path)
        return
    for step_file in step_files:
        step_path = os.path.join(input_folder_path, step_file)
        part_id = os.path.splitext(step_file)[0]
        print(f"Processing {part_id} ...")
        # Load model
        shape = load_step_model(step_path)
        circular_holes = detect_circular_holes(shape)
        # Detect features
        fillets = detect_fillets(
                    shape,
                    min_radius=0.5,      # Minimum fillet radius (mm)
                    max_radius=20.0,     # Maximum fillet radius (mm)
                    classify_type=True   # Distinguish internal/external
                )
        # non_circular_holes = detect_non_circular_holes(shape)
        # wall_thickness = mode_thickness(shape)
        # print(f"Overall wall thickness is {wall_thickness}mm")
        faces = detect_all_faces(shape)
        print(f"Detected {len(faces)} faces")
        # edges = detect_all_edges(shape)
        # Combine and export
        features = {
            "part_id": part_id,
            "circular_holes": circular_holes,
            "fillets": fillets,
            # "edges": edges,
            # "non_circular_holes": non_circular_holes,
            # "wall_thickness": wall_thickness,
            "faces": faces
        }
        out_path = os.path.join(export_path, f"{part_id}_features.json")
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"Exported {out_path}")


def main():
    input_folder_path = "target_steps"
    export_path = "geo_features"
    detect_and_export(input_folder_path, export_path)


if __name__ == "__main__":
    main()