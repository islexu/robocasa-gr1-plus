"""Record simulator/image dependencies that can change reset or rendered pixels."""
from importlib.metadata import version,PackageNotFoundError
import platform

def runtime_versions():
    result={"python":platform.python_version()}
    for name in ("numpy","Pillow","mujoco","gymnasium","opencv-python","opencv-python-headless","robosuite"):
        try: result[name]=version(name)
        except PackageNotFoundError: pass
    return result
