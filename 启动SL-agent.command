#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$PWD/.mplconfig"
exec "$PWD/.venv/bin/lens-agent" ui "$@"
