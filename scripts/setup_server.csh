#!/bin/tcsh -f
setenv UV_TORCH_BACKEND cu129
mkdir -p logs
if (! -d .venv) uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt || exit 1
if (! -d .venv-clef) uv venv --python python3.12 .venv-clef
uv pip install --python .venv-clef/bin/python -r requirements-clef.txt || exit 1
if (! -d .venv-pplx) uv venv --python python3.12 .venv-pplx
uv pip install --python .venv-pplx/bin/python -r requirements-pplx.txt || exit 1
rehash
echo "Setup done."
