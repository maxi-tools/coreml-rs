with open("build.rs") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "rustc-link-search" in line:
        print(f"{i+1}: {line.strip()}")
