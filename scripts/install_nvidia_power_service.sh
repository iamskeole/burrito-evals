#!/bin/bash

# Copy the unit file to systemd
sudo cp nvidia-power.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable --now nvidia-power.service

echo "NVIDIA power limit service installed and started."
