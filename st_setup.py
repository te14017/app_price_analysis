import sys
import subprocess


if __name__ == '__main__':
    if sys.version_info.major < 3:
        action = getattr(subprocess, "call")
    elif sys.version_info.minor < 5:
        action = getattr(subprocess, "call")
    else:
        action = getattr(subprocess, "run")

    action(["conda", "env", "create", "-n", "mp", "-f",
            "st_conda.yml"])

    print("\n========================================================")
    print("Type 'source activate mp' to activate environment.")
    print("==========================================================")
