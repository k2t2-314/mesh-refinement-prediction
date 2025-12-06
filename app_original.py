import gradio as gr
import os
from mesh_service.run_interface import get_result
import gmsh

# gmsh only initialize ONCE
try:
    gmsh.initialize()
except:
    pass


def run_mesh(step_file, thickness, load_type, direction, scale):
    import tempfile

    # 1. Handle file input
    if hasattr(step_file, "read"):
        # HF / Cloud mode
        with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as f:
            f.write(step_file.read())
            step_path = f.name

    elif hasattr(step_file, "name"):
        # Local Gradio mode
        step_path = step_file.name

    else:
        step_path = str(step_file)

    # 2. Convert direction text → number
    if direction == "positive":
        direction_val = 1
    else:
        direction_val = -1

    # 3. Compute
    result = get_result(step_path, float(thickness), load_type, direction_val, scale)

    # 4. Ensure absolute paths for images
    refined_img = os.path.abspath("output/mesh_refined.png")
    global_img = os.path.abspath("output/mesh_global_only.png")

    return result, refined_img, global_img


iface = gr.Interface(
    fn=run_mesh,
    inputs=[
        gr.File(label="Upload STEP File"),
        gr.Textbox(label="Thickness"),
        gr.Dropdown(["bend_bottom", "tension"], value="bend_bottom", label="Load Type"),
        gr.Dropdown(["positive", "negative"], value="positive", label="Direction"),
        gr.Dropdown(["low", "medium", "high"], value="high", label="Load Scale")
    ],
    outputs=[
        gr.Textbox(label="Result", lines=10, max_lines=20),
        gr.Image(label="Refined Mesh"),
        gr.Image(label="Global Mesh")
    ],
    title="Mesh Refinement Agent",
    examples=[
        ["example_steps/example1.step", "3",    "bend_bottom", "positive", "high"],
        ["example_steps/example2.step", "2",    "tension",     "negative", "medium"],
        ["example_steps/example3.step", "3",    "bend_bottom", "positive", "high"],
        ["example_steps/example4.step", "3.5",  "tension",     "negative", "medium"]
    ]
)


iface.launch(server_name="0.0.0.0", server_port=7860)
