from mesh_service import run_interface
import os

def main():
    step_path = "example_steps/example4.step"

    # hickness detection is not available right now
    # Fo the thickness information please input my yourself or refer to the information below:
    # Example1: 3
    # Example2: 2
    # Example3: 3
    # Example4: 3.5

    thickness = 3

    # Input of the load
    load_face = "bend_bottom"
    load_direction = 1
    load_scale = "high"
    run_interface.get_result(step_path, thickness, load_face, load_direction, load_scale)

if __name__ == "__main__":
    main()