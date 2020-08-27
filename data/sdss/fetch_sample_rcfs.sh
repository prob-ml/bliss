#!/bin/bash

xargs -P 50 -n 3 make < sample_rcfs.txt
