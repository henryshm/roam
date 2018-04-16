#!/bin/sh
THISDIR=`pwd`
export PYTHONPATH=$THISDIR/src
cd $THISDIR/notebooks
jupyter notebook
