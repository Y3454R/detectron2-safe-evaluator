import os

def print_tree(startpath, max_files=5):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files[:max_files]:
            print(f"{subindent}{f}")
        if len(files) > max_files:
            print(f"{subindent}... ({len(files) - max_files} more)")

print_tree("/kaggle/input/semantic-segmentation-of-underwater-imagery-suim")
