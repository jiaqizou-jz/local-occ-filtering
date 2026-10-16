#!/bin/bash
# usage: dataset/mujoco/download.sh hopper_gravity_0.5_expert hopper_gravity_2.0_medium ...
# ids come from drive_index.tsv (scraped from the ODRL google drive folder, embeddedfolderview lists all 240 files)
cd "$(dirname "$0")"
for name in "$@"; do
  id=$(awk -v n="$name.hdf5" '$2 == n {print $1}' drive_index.tsv)
  [ -n "$id" ] || { echo "no id for $name"; continue; }
  [ -f "$name.hdf5" ] && { echo "have $name.hdf5"; continue; }
  ~/.local/bin/gdown --id "$id" -O "$name.hdf5"
done
ls -la *.hdf5
