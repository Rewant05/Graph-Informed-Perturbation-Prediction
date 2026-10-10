# Update the existing Windows checkout

Extract the new ZIP under `C:\reproducibility`, giving a folder named `Graph_Perturbation_Study_v2`. Keep the existing checkout and its `.git` directory. Copy the new release contents into it; the ZIP contains no `.git`, datasets or local work checkpoints.

```powershell
Copy-Item "C:\reproducibility\Graph_Perturbation_Study_v2\*" "C:\reproducibility\Graph_Perturbation_Study" -Recurse -Force
cd C:\reproducibility\Graph_Perturbation_Study
python .\verify_release.py
git config core.autocrlf false
git add --renormalize .
git add .
git status --short
git commit -m "Add model entry points, preprocessing and Sphinx documentation"
git push origin main
```

Review `git status` before committing. The new `.gitattributes` preserves exact release-hashed bytes. Re-staging with `--renormalize` applies it to already tracked files; `git add .` also stages new files. Local `work/` paths remain ignored. Documentation hosting requires importing the repository through your Read the Docs account.
