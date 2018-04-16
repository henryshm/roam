#!/bin/sh
export PYTHONPATH=`pwd`/src
python3 -m pytest src
