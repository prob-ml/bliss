#!/bin/bash

for i in {12..13}; do echo "RUN=94 CAMCOL=1 FIELD=$i"; done  > rcf_list
xargs -P 50 -n 3 make < rcf_list
xargs -P 50 -n 3 make photoobj < rcf_list
xargs -P 50 -n 3 make psfield < rcf_list
