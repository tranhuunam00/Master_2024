#!/bin/bash
# SiPAP air-path CFD: mesh once, then simpleFoam at 60 / 30 / 80 L/min.
# Run inside WSL Ubuntu with the Ubuntu package `openfoam` (ESI v1912):
#   bash run_all.sh <folder with gen_case.py + geom/airpath.stl + geom/airpath.json>
# Cases go to ~/sipap_cfd (Linux disk: much faster than /mnt/c).
set -e
SRC=${1:-$(dirname "$(readlink -f "$0")")}
export WM_PROJECT_DIR=/usr/share/openfoam FOAM_ETC=/usr/share/openfoam/etc WM_PROJECT=OpenFOAM WM_PROJECT_VERSION=v1912

C=~/sipap_cfd/mesh
rm -rf ~/sipap_cfd; mkdir -p $C/constant/triSurface
python3 $SRC/gen_case.py $C $SRC/geom/airpath.json 60
cp $SRC/geom/airpath.stl $C/constant/triSurface/
cd $C
blockMesh > log.blockMesh 2>&1
surfaceFeatureExtract > log.sfe 2>&1
snappyHexMesh -overwrite > log.snappy 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Mesh OK|Failed" log.checkMesh

set +e
for Q in 60 30 80; do
  C=~/sipap_cfd/q$Q
  rm -rf $C; mkdir -p $C
  python3 $SRC/gen_case.py $C $SRC/geom/airpath.json $Q
  cp -r ~/sipap_cfd/mesh/constant/polyMesh $C/constant/
  cd $C
  decomposePar -force > log.decompose 2>&1
  mpirun --oversubscribe -np 8 simpleFoam -parallel > log.simpleFoam 2>&1
  reconstructPar -latestTime > log.reconstruct 2>&1
  foamFormatConvert -latestTime > log.convert 2>&1
  foamFormatConvert -constant > log.convert2 2>&1
  echo "q$Q done" >> ~/sipap_cfd/status.txt
done
echo ALL_DONE >> ~/sipap_cfd/status.txt
# then on Windows: python post.py \\wsl.localhost\Ubuntu\home\<user>\sipap_cfd geom\airpath.json results
