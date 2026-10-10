# Build and host the documentation

The source includes Sphinx/MyST pages, a Read the Docs theme and `.readthedocs.yaml`. Local HTML builds do not require PyTorch or importing the model modules.

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs work/docs-html
```

Open `work/docs-html/index.html` locally.

To publish: push the source/configuration to GitHub, sign in to Read the Docs, import `Rewant05/Graph-Informed-Perturbation-Prediction`, select its main branch and trigger a build. Use the site URL assigned by that service; do not claim a hosted site exists before a successful build. Only documentation dependencies are installed for this build.

References:

- https://docs.readthedocs.com/platform/stable/config-file/v2.html
- https://nichecompass.readthedocs.io/en/latest/
- https://scanpy.readthedocs.io/en/stable/generated/scanpy.external.pp.harmony_integrate.html
