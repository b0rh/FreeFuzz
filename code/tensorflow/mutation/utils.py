from genericpath import exists
import os
import subprocess

def code_to_file(code: str, filename: str, output_dir: str, dump_import: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename + ".py"), "w") as f:
        if dump_import:
            f.write("import tensorflow as tf\n")
            f.write("import numpy as np\n")
        f.write(code)


def run_code(code: str, filename: str, output_dir: str):
    code_to_file(code, filename, output_dir)
    py = "python3"
    return subprocess.run([py, os.path.join(output_dir, filename+".py")]).returncode
