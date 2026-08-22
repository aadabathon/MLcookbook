"""MLcookbook desktop GUI layer — dumb presentation only.

Widgets here render data they are handed and emit signals; they never import
from mlcookbook (the analytical core) and never decide navigation. The
repo-root app.py is the single seam that knows both sides.

Launch the app with:  python -m mlcookbook
"""
