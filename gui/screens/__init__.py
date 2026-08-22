"""Recipe screens — one module per recipe view.

Screens are dumb: they render what app.py hands them (plain strings/dicts),
emit ``back_requested`` to leave, and never import from mlcookbook themselves.
"""
