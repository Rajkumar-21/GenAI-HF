#!/bin/bash

# Ensure necessary tools are installed
# sudo apt-get update
# sudo apt-get install -y wget tar build-essential

# Navigate to a directory for SQLite3
cd ~/sqlite3

# Change into the extracted directory
cd sqlite-autoconf-3490100

# Configure, build, and install SQLite3
./configure
make
sudo make install

# Verify the installation
sqlite3 --version
