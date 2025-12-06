#!/bin/bash

# Define variables
#MOUNT_DIR="/mnt/data"
#DISK_NAME="YOUR_DISK_DEVICE_NAME" # Often something like 'sdb' or 'sdf'

# NB: Not using actual GCP username here : 
#   Could check with : `gcloud compute os-login describe-profile` (but didn't want to...)
TPU_USER=tpu_user

JUPYTER_USER=${TPU_USER}
JUPYTER_PORT=8585


# Ensure the user exists (replace 'your_username' with the actual username)
if ! id ${TPU_USER} &>/dev/null; then
  useradd -m ${TPU_USER}
fi


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


## Install a package as 'your_username'
#sudo -u your_username bash -c 'pip install my-package'

## Run a script as 'your_username'
#sudo -u your_username /home/your_username/myscript.sh


# Switch to 'your_username' and execute commands
sudo -u ${TPU_USER} bash << EOF
  cd /home/${TPU_USER}
  #whoami

  pip freeze | sort > 0-pip-freeze.log

  # Install Jupyter and necessary packages (if not already in your VM image)
  pip install jupyter jupyterlab jupytext --user

  #  WARNING: The scripts jlpm, jupyter-lab, jupyter-labextension and jupyter-labhub 
  #    are installed in '/home/tpu_user/.local/bin' which is not on PATH.
  #  Consider adding this directory to PATH 
  #    or, if you prefer to suppress this warning, use --no-warn-script-location.

  # Start JupyterLab server in the background as the user
  # Use --no-browser and --ip=0.0.0.0 to make it accessible remotely
  # Use nohup to keep it running after the script finishes
  nohup .local/bin/jupyter lab --no-browser --ip=0.0.0.0 --port=${JUPYTER_PORT} --allow-root &


  pip install numpy
  pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip freeze | sort > 1-pip-freeze_with_jax.log
  # JAX version: 0.6.2  ??

  pip install -q git+https://github.com/google/tunix
  pip freeze | sort > 2-pip-freeze_with_tunix.log

  pip install -q git+https://github.com/google/qwix  
  pip freeze | sort > 3-pip-freeze_with_qwix.log

  #pip uninstall -q -y flax  # No need here - flax not installed yet
  pip install --no-cache-dir git+https://github.com/google/flax.git
  pip freeze | sort > 4-pip-freeze_with_flax.log



  # https://docs.cloud.google.com/compute/docs/instances/startup-scripts/linux#accessing-metadata
  #   confirmed : this does get passed to metadata store for downloading to 0-metadata.txt
  METADATA_FOO_VALUE=\$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/foo -H "Metadata-Flavor: Google")
  echo \${METADATA_FOO_VALUE} > 0-metadata.txt

EOF

# Does nothing ( just here as a placeholder / comment )
cat << EOF
  
EOF


