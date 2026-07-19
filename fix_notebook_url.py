"""One-shot script to patch the GitHub URL in the Colab template notebook."""
import json
import pathlib

nb_path = pathlib.Path(
    r"c:/Users/SAKSHI VERMA/Documents/Internship 2/resonova/notebooks/resonova_colab_template.ipynb"
)
nb = json.loads(nb_path.read_text(encoding="utf-8"))

fixed = 0
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        new_src = []
        for line in cell["source"]:
            if "SAK_SHI14" in line and "GITHUB_REPO_URL" in line:
                line = "GITHUB_REPO_URL = 'https://github.com/SAK-SHI14/Resonova.git'  # ← your GitHub URL\n"
                fixed += 1
            new_src.append(line)
        cell["source"] = new_src

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Fixed {fixed} URL(s). Notebook saved.")
