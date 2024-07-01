import os
import launch

if not launch.is_installed("triton"):
    if os.name == 'nt':
        launch.run_pip(
            f"install https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl",
            "UDAV2 install triton from custom wheel"
        )
    else:
        launch.run_pip(f"install triton", "UDAV2 install triton")
