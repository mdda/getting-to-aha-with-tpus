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

apt-get update -y 
apt install python3.12 python3.12-venv -y 


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

  mkdir -p ~/.kaggle/

  python3.12 -m venv ./env-tpu
  source ./env-tpu/bin/activate

  pip install -U pip
  pip freeze | sort > 0-pip-freeze.log  # NOTHING!


  # Install Jupyter and necessary packages
  pip install jupyter jupyterlab jupytext
  pip freeze | sort > 1-pip-freeze_with_jupyter.log

  # Start JupyterLab server in the background as the user
  #   Use --no-browser and --ip=0.0.0.0 to make it accessible remotely
  #   Use nohup to keep it running after the script finishes
  nohup jupyter lab --no-browser --ip=0.0.0.0 --port=${JUPYTER_PORT} --allow-root &


  pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  # 0.6.2 (with Python 3.10 installed == BAD CHOICE)
  # 0.8.1 (with Python 3.12 installed)
  pip freeze | sort > 2-pip-freeze_with_jax.log

  #pip install --no-cache-dir git+https://github.com/google/flax.git
  pip install git+https://github.com/google/flax.git
  pip freeze | sort > 3-pip-freeze_with_flax.log

  pip install git+https://github.com/google/tunix git+https://github.com/google/qwix
  pip freeze | sort > 4-pip-freeze_with_tunix-qwix.log
  # This one also gives us kagglehub!


  # https://docs.cloud.google.com/compute/docs/instances/startup-scripts/linux#accessing-metadata
  #   WORKS confirmed : this does get passed to metadata store for downloading to 0-metadata.txt
  METADATA_FOO_VALUE=\$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/foo -H "Metadata-Flavor: Google")
  echo \${METADATA_FOO_VALUE} > 0-metadata.txt

  # Interesting:
  # curl http://metadata.google.internal/computeMetadata/v1/instance/ -H "Metadata-Flavor: Google"
  #   gives us back a bunch of information in (apparently) a nice structure
  # eg: .../instance/machine-type -> "projects/714NNNNNNN0/machineTypes/n2d-48-24-v5lite-tpu"

EOF

# Does nothing ( just here as a placeholder / comment )
cat << EOF
  
EOF


