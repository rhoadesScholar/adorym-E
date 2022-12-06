#!/bin/bash
conda create -n adorym-E python=3.9 pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
conda activate adorym-E
pip install -e .

