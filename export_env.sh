#!/bin/bash
echo "~~ exporting environment spec... ~~"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
conda env export -p $PWD/miniconda --file environment.yml
