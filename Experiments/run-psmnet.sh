#!/bin/bash
PY3=python3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR=/cs231a/data/kitti
DATE=2011_09_26
DRIVE=0001
WEIGHTS=/cs231a/psmnet-pretrained_model_KITTI2015.tar

# echo 'run.sh is using: '
# echo " - BASE_DIR = ${BASE_DIR}"
# echo " - DATE = ${DATE}"
# echo " - DRIVE = ${DRIVE}"
# echo " - WEIGHTS = ${WEIGHTS}"
# echo

CMD="${PY3} ${SCRIPT_DIR}/pykitti-psmnet.py --base_dir ${BASE_DIR} \
                                            --date ${DATE} \
                                            --drive ${DRIVE} \
                                            --pretrained_weights_path ${WEIGHTS}"
echo "Running:"
echo ${CMD}
${CMD}
