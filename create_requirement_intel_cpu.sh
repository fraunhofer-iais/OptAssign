#!/usr/bin/env bash

set -eu


echo "--extra-index-url https://download.pytorch.org/whl/cpu" > requirements_intel_cpu.txt
uv pip freeze >> requirements_intel_cpu.txt

