from __future__ import annotations
import argparse, json
from .ply_io import inspect

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    args = p.parse_args()
    print(json.dumps(inspect(args.ply), indent=2))
