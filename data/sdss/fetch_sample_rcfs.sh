#!/bin/bash

for i in {16..830}; do echo "RUN=3900 CAMCOL=6 FIELD=$i"; done  > rcf_list
xargs -P 50 -n 3 make < rcf_list
xargs -P 50 -n 3 make photoobj < rcf_list
xargs -P 50 -n 3 make psfield < rcf_list
