#!/usr/bin/env bash
# Official Zenodo v1 archive. Retain the tar for checksum-based provenance.
set -euo pipefail
mkdir -p data
if [[ ! -f data/CIFAR-100-C.tar ]]; then
  curl --fail --location --retry 3 --continue-at - --output data/CIFAR-100-C.tar.part https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
  printf '%s\n' '11f0ed0f1191edbf9fa23466ae6021d3  data/CIFAR-100-C.tar.part' | md5sum --check -
  mv data/CIFAR-100-C.tar.part data/CIFAR-100-C.tar
fi
printf '%s\n' '11f0ed0f1191edbf9fa23466ae6021d3  data/CIFAR-100-C.tar' | md5sum --check -
# Skip complete extractions; rerunning an interrupted extraction only restores archive files.
if [[ ! -f data/CIFAR-100-C/.extracted-v1 ]]; then
  tar -xf data/CIFAR-100-C.tar -C data
  touch data/CIFAR-100-C/.extracted-v1
fi
python3 -m scripts.check_cifar100_data
