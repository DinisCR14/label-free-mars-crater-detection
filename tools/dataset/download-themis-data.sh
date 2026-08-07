#!/usr/bin/env bash
set -euo pipefail

# Download source assets into an external directory, never into the repository.
# Review the source terms and available endpoints before running this script.

output_dir="${1:?Usage: $0 <external-data-directory>}"
mkdir -p "$output_dir/raw/thm" "$output_dir/raw/thmhd"

wget -r -np -nH -R 'index.html*' -e robots=off -N \
  -P "$output_dir/raw/thm" \
  'https://www.mars.asu.edu/data/thm_dir/large/'

wget -r -np -nH -R 'index.html*' -e robots=off -N \
  -P "$output_dir/raw/thmhd" \
  'http://www.mars.asu.edu/data/thm_dir_100m/large/'
