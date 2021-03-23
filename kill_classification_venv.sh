#!/usr/bin/env bash

VENVNAME=classification
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME