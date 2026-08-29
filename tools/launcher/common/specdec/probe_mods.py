import sys
print("exe", sys.executable)
for name in ("pyarrow", "pandas", "datasets", "numpy", "torch"):
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "?"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__)
