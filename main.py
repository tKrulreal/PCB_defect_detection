"""PCB Defect Detection System — Main Entry Point.

Usage:
    python main.py demo          Run quick demo (images from demo_input/)
    python main.py train         Train Stage 2 CNN (see --help for options)
    python main.py evaluate      Evaluate full YOLO + CNN pipeline
    python main.py visualize     Generate result visualizations
    python main.py gradcam       Generate GradCAM heatmaps
    python main.py compare       Compare Stage 2 models
    python main.py app           Launch Streamlit web app

Examples:
    python main.py demo --input path/to/image.jpg
    python main.py train --model resnet18 --epochs 50
    python main.py evaluate --split test --cnn runs/stage2/resnet18/best.pt
    python main.py visualize --model all
    python main.py gradcam --input demo_input/ --cnn runs/stage2/resnet18/best.pt
    python main.py app
"""

import subprocess
import sys


COMMANDS = {
    "demo": {
        "script": ["demo_stage12_one_image.py"],
        "desc": "Run quick demo on images",
    },
    "train": {
        "script": ["stage2_train.py"],
        "desc": "Train Stage 2 CNN classifier",
    },
    "evaluate": {
        "script": ["evaluate_stage12_system.py"],
        "desc": "Evaluate full YOLO + CNN pipeline",
    },
    "visualize": {
        "script": ["visualize_results.py"],
        "desc": "Generate result visualizations",
    },
    "gradcam": {
        "script": ["gradcam_visualize.py"],
        "desc": "Generate GradCAM heatmaps",
    },
    "compare": {
        "script": ["compare_stage2_models.py"],
        "desc": "Compare Stage 2 CNN models",
    },
    "app": {
        "script": ["-m", "streamlit", "run", "app.py"],
        "desc": "Launch Streamlit web application",
    },
}


def print_usage():
    """Print available commands with descriptions."""
    print(__doc__)
    print("Available commands:")
    for name, info in COMMANDS.items():
        print(f"  {name:<14} {info['desc']}")
    print()


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1]

    if command in ("--help", "-h"):
        print_usage()
        sys.exit(0)

    if command not in COMMANDS:
        print(f"Error: unknown command '{command}'\n")
        print_usage()
        sys.exit(1)

    remaining_args = sys.argv[2:]
    script_parts = COMMANDS[command]["script"]
    full_command = [sys.executable] + script_parts + remaining_args

    try:
        result = subprocess.run(full_command, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
