"""Serialize Ray runtime settings without interpolating values into JSON code."""

from __future__ import annotations

import argparse
import json


def runtime_environment(assignments, working_dir=None):
    environment = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise ValueError("Runtime environment values must use KEY=VALUE")
        environment[key] = value
    result = {"env_vars": environment}
    if working_dir is not None:
        result["working_dir"] = working_dir
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--working-dir")
    arguments = parser.parse_args()
    print(json.dumps(runtime_environment(arguments.env, arguments.working_dir)))


if __name__ == "__main__":
    main()
