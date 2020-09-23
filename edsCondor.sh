#!/usr/bin/env bash
python create_dag.py
python create_sub_files.py
mkdir outfiles
condor_submit_dag eds.dag
