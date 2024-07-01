import os
import launch

extension_dir = os.path.dirname(os.path.realpath(__file__))

dependencies = [
    ("gradio", "gradio"),
    ("safetensors", "safetensors"),
    ("matplotlib", "matplotlib"),
    ("opencv-python", "opencv-python"),
    ("argparse", "argparse")
]

for package, name in dependencies:
    if not launch.is_installed(name):
        launch.run_pip(f"install {package}", package)

def download_file(url, dest):
    if not os.path.exists(dest):
        launch.run(f"curl -L -o {dest} {url}")

checkpoints = [
    ("https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl?download=true", os.path.join(extension_dir, "triton-2.1.0-cp310-cp310-win_amd64.whl"))
]

for url, dest in checkpoints:
    download_file(url, dest)

triton_wheel = os.path.join(extension_dir, "triton-2.1.0-cp310-cp310-win_amd64.whl")
if not launch.is_installed("triton"):
    if os.path.exists(triton_wheel):
        launch.run_pip(f"install {triton_wheel}", "triton")