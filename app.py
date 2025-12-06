## MVP 1 FILE!
import gradio as gr
import os
import json
import tempfile
import gmsh

from mesh_service.run_interface import (
    get_result,
    extract_faces_for_gui   # <-- NEW helper from updated run_interface
)

# Absolute path to example files inside Docker container
os.chdir(os.path.dirname(__file__))
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "example_steps")


# -----------------------------------------
# Gmsh initialize only once
# -----------------------------------------
try:
    gmsh.initialize()
except:
    pass


# -----------------------------------------
# STEP Upload → Extract Faces for GUI
# -----------------------------------------
def on_step_upload(step_file):
    if step_file is None:
        return gr.update(choices=[], value=None), "[]"

    # Work in temp copy if needed
    if hasattr(step_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as f:
            f.write(step_file.read())
            step_path = f.name
    else:
        step_path = step_file.name

    faces = extract_faces_for_gui(step_path)
    labels = [f["label"] for f in faces]

    faces_json = json.dumps(faces)
    #default = labels[0] if labels else None
    default = None

    return gr.update(choices=labels, value=default), faces_json


# -----------------------------------------
# Main pipeline call
# -----------------------------------------
def run_mesh(step_file, thickness, load_type, direction, scale,
             selected_face_label, faces_json, view_mode):
    
    # 1. Handle STEP file path
    if hasattr(step_file, "read"):
        # Cloud / HuggingFace mode
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as f:
            f.write(step_file.read())
            step_path = f.name
    else:
        step_path = step_file.name

    # 2. Convert direction text → number
    direction_val = 1 if direction == "positive" else -1

    # 3. Run model pipeline
    sentence, global_png, refined_png, fig, msh_path = get_result(
        step_path=step_path,
        thickness=float(thickness),
        load_face=load_type,     # still used as categorical feature
        load_direction=direction_val,
        load_scale=scale,
        selected_face_label=selected_face_label,  # NEW
        faces_json=faces_json,                    # NEW
        view_mode=view_mode
    )

    # 4. Convert PNG paths to absolute (for Gradio)
    refined_img = os.path.abspath(refined_png)
    global_img  = os.path.abspath(global_png)
    msh_path = os.path.abspath(msh_path)

    return sentence, refined_img, global_img, fig, msh_path


# -----------------------------------------
# Gradio Interface
# -----------------------------------------
with gr.Blocks(title="Mesh Refinement Agent") as iface:
    
    gr.Markdown("## Mesh Refinement Prediction Demo")

    gr.HTML("""
<details style="margin-bottom: 18px;">
  <summary style="font-size: 18px; cursor: pointer; font-weight: bold;">
    How to Choose the Correct Load Face
  </summary>
  <div style="padding-left: 12px; margin-top: 10px;">

  <p>Different load types correspond to different physical loading scenarios.<br>
  After uploading your STEP file, the face dropdown will list <b>all detected faces</b>.<br>
  To ensure the ML model receives inputs aligned with its training distribution,
  please choose the face following these guidelines:</p>

  <h3><b>Load Type: bend_bottom</b></h3>
  <p>Use this when the bracket or part is <b>loaded from below</b>, such as:</p>
  <ul>
    <li>A shelf bracket being pressed upward</li>
    <li>A support arm being bent from underneath</li>
  </ul>

  <p><b>Choose the face that is physically on the bottom of the part</b>, i.e.:</p>
  <ul>
    <li>The largest downward-facing surface</li>
    <li>The surface that would contact a support or wall in real usage</li>
  </ul>

  <p>💡 <i>Tip:</i> In most STEP files this is a face with a downward normal  
  (<code>normal ≈ [0, 0, -1]</code>).</p>

  <hr>

  <h3><b>Load Type: tension</b></h3>
  <p>Use this when the bracket or part is pulled <b>backwards</b> or <b>outwards</b>, such as:</p>
  <ul>
    <li>A wall bracket being pulled away from a mounting plane</li>
    <li>A hook experiencing outward tension</li>
  </ul>

  <p><b>Choose the face that is physically the rear/back face of the part</b>, i.e.:</p>
  <ul>
    <li>The flat face that would mount to a wall</li>
    <li>A large face with outward-pointing orientation</li>
  </ul>

  <p>💡 <i>Tip:</i> Often the normal points along ±X  
  (<code>normal ≈ [±1, 0, 0]</code>).</p>

  <hr>

  <h3><b>If You Are Unsure</b></h3>
  <ul>
    <li>Choose the face that best represents the real-life load direction.</li>
    <li>Picking an incorrect face has <b>minimal effect</b> on the mesh output.</li>
  </ul>

  </div>
</details>
""")

    gr.HTML("""
<details style="margin-bottom: 18px;">
  <summary style="font-size: 18px; cursor: pointer; font-weight: bold;">
    3D Visualization Mode Guidance
  </summary>
  <div style="padding-left: 12px; margin-top: 10px;">

  <p>In the 3D visualization mode dropdown, there are three modes to choose from:</p>

  <ul>
    <li><b>wireframe</b>: 3D interactive wireframe of the final optimized mesh</li>
    <li><b>highlight</b>: wireframe with refined regions shown in red</li>
    <li><b>heatmap</b>: continuous heatmap quantifying refinement magnitude</li>
  </ul>

  <p><b>Heatmap color scale:</b></p>
  <ul>
    <li><b>Red</b>: very small predicted mesh size → highest refinement</li>
    <li><b>Blue</b>: very large predicted mesh size → coarsest regions</li>
  </ul>

  <p>The heatmap provides a smooth, global visualization of refinement intensity across the mesh.</p>

  </div>
</details>
""")

    
    with gr.Row():
        with gr.Column(scale=1):
            
            # STEP file upload
            step_file = gr.File(
                label="Upload STEP File",
                file_types=[".step", ".stp"]
            )

            thickness = gr.Textbox(
                label="Thickness (mm)",
                value="3"
            )

            load_type = gr.Dropdown(
                ["bend_bottom", "tension"],
                value="bend_bottom",
                label="Load Type (categorical feature)"
            )

            direction = gr.Dropdown(
                ["positive", "negative"],
                value="positive",
                label="Direction"
            )

            scale = gr.Dropdown(
                ["low", "medium", "high"],
                value="high",
                label="Load Scale"
            )

            # toggle visualization mode
            view_mode = gr.Dropdown(
                ["wireframe", "highlight", "heatmap"],
                value="wireframe",
                label="3D Visualization Mode"
            )

            # NEW: face selection & json hidden box
            selected_face_label = gr.Dropdown(
                label="Select Load Face (detected from STEP)",
                choices=[],
                value=None,
                interactive=True
            )

            faces_json_box = gr.Textbox(
                visible=False
            )

            examples_data = [
                [f"{EXAMPLE_DIR}/example1.step", "3", "bend_bottom", "positive", "high", None, "[]", "wireframe"],
                [f"{EXAMPLE_DIR}/example2.step", "2", "tension", "negative", "medium", None, "[]", "highlight"],
                [f"{EXAMPLE_DIR}/example3.step", "3", "bend_bottom", "positive", "high", None, "[]", "heatmap"],
                [f"{EXAMPLE_DIR}/example4.step", "3.5", "tension", "negative", "medium", None, "[]", "wireframe"],
            ]


            # Populate face dropdown upon STEP upload
            step_file.change(
                on_step_upload,
                inputs=step_file,
                outputs=[selected_face_label, faces_json_box]
            )

            run_button = gr.Button("Run Mesh Prediction")

        with gr.Column(scale=1):

            result_text = gr.Textbox(
                label="Result",
                lines=12,
                max_lines=20,
                interactive=False
            )

            refined_img = gr.Image(
                label="Refined Mesh (PNG)"
            )

            global_img = gr.Image(
                label="Global Mesh (PNG)"
            )

            # NEW: 3D interactive refined mesh
            mesh_plot_3d = gr.Plot(
                label="Refined Mesh (3D Interactive)"
            )

            mesh_download = gr.File(
                label="Download Generated Mesh (.msh)",
                interactive=False
            )

    gr.Examples(
        examples=examples_data,
        inputs=[
            step_file, thickness, load_type, direction, scale, 
            selected_face_label, faces_json_box, view_mode
        ],
        # The `run_on_click` parameter automatically runs the main function (run_mesh)
        # when an example is selected, giving immediate results.
        # If you prefer users click the 'Run' button manually after selection, set this to False.
        fn=run_mesh,
        outputs=[
            result_text,
            refined_img,
            global_img,
            mesh_plot_3d,
            mesh_download
        ],
        run_on_click=False 
    )

    # Connect compute button
    run_button.click(
        run_mesh,
        inputs=[
            step_file,
            thickness,
            load_type,
            direction,
            scale,
            selected_face_label,
            faces_json_box,
            view_mode,
        ],
        outputs=[
            result_text,
            refined_img,
            global_img,
            mesh_plot_3d,
            mesh_download
        ]
    )


iface.launch(server_name="0.0.0.0", server_port=7860)
