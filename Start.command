#!/usr/bin/env sh
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$task_root/Start.sh"
