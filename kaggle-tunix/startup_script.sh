#!/bin/bash

# Define variables
#MOUNT_DIR="/mnt/data"
#DISK_NAME="YOUR_DISK_DEVICE_NAME" # Often something like 'sdb' or 'sdf'


# Your actual GCP username : Check with : `gcloud compute os-login describe-profile`
JUPYTER_USER="YOUR_USERNAME" 

JUPYTER_PORT=8585

## Check if the disk is already mounted (optional but good practice)
#if ! mountpoint -q "$MOUNT_DIR"; then
#  # Create mount directory if it doesn't exist
#  mkdir -p "$MOUNT_DIR"
#
#  # Format the disk if not already formatted (use 'lsblk -f' to check first)
#  # WARNING: This erases all data on the disk. Uncomment only if necessary.
#  # mkfs.ext4 /dev/disk/by-id/google-$DISK_NAME
#
#  # Mount the persistent disk
#  mount /dev/disk/by-id/google-"$DISK_NAME" "$MOUNT_DIR"
#  # Optional: Add to fstab for more robust automatic mounting on future reboots
#  echo "/dev/disk/by-id/google-$DISK_NAME $MOUNT_DIR ext4 defaults,nofail 0 2" | tee -a /etc/fstab
#
#  # Change ownership to the desired user so Jupyter can access files
#  chown -R "$JUPYTER_USER":"$JUPYTER_USER" "$MOUNT_DIR"
#fi

# Install Jupyter and necessary packages (if not already in your VM image)
# Run as the user

su - "$JUPYTER_USER" -c "pip install jupyter jupyterlab --user"

# Start JupyterLab server in the background as the user
# Use --no-browser and --ip=0.0.0.0 to make it accessible remotely
# Use nohup to keep it running after the script finishes
su - "$JUPYTER_USER" -c "nohup jupyter lab --no-browser --ip=0.0.0.0 --port=$JUPYTER_PORT --allow-root &"

