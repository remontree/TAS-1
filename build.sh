#!/usr/bin/env bash
set -o errexit  # 에러 발생 시 즉시 중단

# 1) 시스템 패키지 설치
apt-get update -y
apt-get install -y gfortran build-essential python3-dev pkg-config

# 2) pip 업그레이드 + Python 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
