#!/usr/bin/env bash
apt-get update
apt-get install -y libzbar0 libzbar-dev python3-dev
pip install -r requirements.txt
