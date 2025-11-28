# MLP-Mesh-Generator

A machine-learning–guided mesh refinement framework for STEP geometries.  
The system automatically detects geometric features, predicts global/local mesh sizes using four MLP models, and generates refined Gmsh meshes with visualization.

---

## Project Structure

MLP-Mesh-Generator/
│
├── example_steps/ # Sample STEP files (sourced from McMaster-Carr)
├── mesh_service/
│ ├── step_feature_detection.py # Feature extraction from STEP geometry
│ ├── run_detection.py # Batch feature detection interface
│ ├── run_ansys.py # Heavy Ansys PyMechanical script (used ONLY for dataset collection)
│ ├── run_interface.py # Main inference pipeline (feature → prediction → meshing → output)
│ ├── model_train.py # Training scripts for all 4 output models
│ ├── part_level/ # Part-level dataset
│ ├── geo_features/ # Feature-level dataset
│ ├── results/ # Prediction results
│ ├── target_steps/ # Target steps for detection
│ └── init.py
│
├── saved_model/ # Trained PyTorch models and preprocessors
├── output/ # Generated meshes (.png + .msh)
├── runtime.py # Main user entry point
├── requirements.txt
└── README.md

---

## Important Notes

### 1. `run_ansys.py` should NOT be executed
This script uses **Ansys PyMechanical** to generate FEA results for 300 loading conditions × 25 STEP parts.  
It is extremely slow and only included for completeness.  
All processed results are already stored in:

- `mesh_service/geo_features/feature_level_augmented.xlsx`
- `mesh_service/part_level/part_level.xlsx`

So you do **not** need Ansys to run the project.

---

## Dataset Description

### STEP Geometry Source
All geometries come from publicly available STEP files on **McMaster-Carr**.  
Only **corner-type parts** were used in the dataset.

### FEA Data Collection
Using `run_ansys.py` with PyMechanical:

- 25 parts  
- 300 loading conditions each  
- Total: 7500 simulations  
- All results distilled into the two Excel datasets above  
- Dataset is intentionally narrow-domain, high-quality, small-sample

### Geometric Features Extracted
`step_feature_detection.py` extracts:

- planar faces  
- circular holes  
- non-circular holes  
- fillets  
- thickness  
- edge and face adjacency  
- center coordinates + normals  
- feature radii  

---

## Model Training

Four separate MLP models are trained (stored in `saved_model/`):

1. **Model 1 — Global mesh size**
2. **Model 2 — Convergence classifier**
3. **Model 3 — Local refinement flag**
4. **Model 4 — Local mesh size**

Training script: `mesh_service/model_train.py`

Each model has:

- Preprocessor (scikit-learn ColumnTransformer)
- PyTorch MLP
- 5-fold validation
- Stored checkpoints and scalers

---

## Model Performance Evaluation

This project trains **four independent MLP models**, each responsible for a different stage of the mesh refinement pipeline.  
Below is a summary of their performance, using cross-validation and final evaluation metrics.

---

### **Model 1 — Global Mesh Size Regression**
- Predicts the global element size for the entire part.
- Training curves show stable convergence.
- Final test performance:
  - **MSE: 1.115**
  - **MAE: 0.253**
- The model captures overall part-scale geometry effectively and avoids collapsing to the mean.

---

### **Model 2 — Convergence Classification**
- Predicts whether an FEA simulation converges under a given global mesh size.
- Cross-validation accuracy: **0.58 ± 0.06**
- Final model on full dataset:
  - **Accuracy: 0.82**
  - **Precision (class 1): 0.91**
  - **Recall (class 1): 0.82**
  - **F1-score: 0.87**
- The classifier shows strong predictive ability for convergence, even with imbalanced labels.

---

### **Model 3 — Local Refinement Flag**
- Predicts whether a given face requires local refinement.
- Validation performance:
  - **Best threshold (Youden’s J): 0.1588**
  - **TPR: 0.9825**, **FPR: 0.3740**
- Test metrics:
  - **MSE: 0.116**
  - **MAE: 0.253**
- Confusion matrix reveals excellent detection of positive refinement regions (high recall), which is desirable for downstream mesh selection.

---

### **Model 4 — Local Mesh Size Regression**
- Predicts the refined element size for high-stress faces.
- 5-fold cross-validation:
  - **Mean MSE: 0.5135 ± 0.2575**
  - **Mean MAE: 0.5517 ± 0.1594**
- Final model trained on all data:
  - **MSE: 0.2023**
  - **MAE: 0.3642**
- The model consistently reduces prediction error after 300–400 epochs and demonstrates reliable refinement scale estimation.

---

### Summary
| Model | Task | Metric | Score |
|-------|------|--------|--------|
| Model 1 | Global mesh size regression | MAE | **0.253** |
| Model 2 | Convergence classification | Accuracy | **0.82** |
| Model 3 | Local refine flag | MAE | **0.253** |
| Model 4 | Local mesh size regression | MAE | **0.364** |

These four models jointly form a robust ML pipeline capable of performing end-to-end mesh size prediction for both global and local refinement regions.

---

## Usage

### 1. Install dependencies
`pip install -r requirements.txt`

### 2. Run the main interface
`python runtime.py`

### 3. Example configuration inside `runtime.py`
`step_path = "mesh_service/example_steps/example4.step"
thickness = 3
load_face = "bend_bottom"
load_direction = 1
load_scale = "high"`

### 4. Output files (in `output/`)

- `mesh_global_only.png` — global mesh  
- `mesh_refined.png` — refined mesh (global + ML local refinement)  
- `mesh_refined.msh` — Gmsh mesh file  
- `inference_results.json` — model outputs

---

## Output Description

`mesh_refined.png`
- Shows final mesh  
- Red-highlighted faces = refined regions  
- Local mesh sizes override global size (via Gmsh field)

`mesh_refined.msh`
- Usable directly in FEA

`inference_results.json`
- Global size
- Convergence prediction
- Top-2 high-stress faces
- Local refinement decisions + predicted sizes

---

## License
MIT License

---

## Contributors
- Xinxuan Tang (CMU Mechanical Engineering)
