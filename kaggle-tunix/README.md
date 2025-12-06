## kaggle-tunix notes

### GCP set-up

* [TPU pricing](https://cloud.google.com/tpu/pricing)
* [Where are the TPUs?](https://docs.cloud.google.com/tpu/docs/regions-zones)
* [TPU Software versions](https://docs.cloud.google.com/tpu/docs/runtimes)

```bash
# gcloud auth login
# gcloud config set project <PROJECT_ID>
# gcloud services enable tpu.googleapis.com

# Quotas (under IAM & Admin / Quotas):
#   Cloud TPU API : "tpu" "v5" : Apparently 32 everywhere!

# See: https://docs.cloud.google.com/tpu/docs/setup-gcp-account
#   Add essential IAM Roles to <YOUR_GCP_USER> :
#     Your account needs specific permissions to create and manage TPUs. The most critical role to ADD is TPU Admin.
#       Service Account Admin: Needed to create a service account
#       Project IAM Admin: Needed to grant a role in a project
#       TPU Admin: Needed to create a TPU
#   Create a Cloud TPU service agent:
#     `gcloud beta services identity create --service tpu.googleapis.com`  #  --project PROJECT_ID

# NOT DONE (YET):
#   You might also find these roles helpful or necessary, depending on your workflow:
#     Service Account Admin ( roles/iam.serviceAccountAdmin ): 
#       If you plan to use a custom service account for your TPU VM, this role allows you to manage service accounts.
#     Storage Admin ( roles/storage.admin ): 
#       Cloud TPUs often interact with Cloud Storage for data and model checkpoints, 
#       so this is generally a good permission to have for seamless operations.

# NOT DONE (YET):
#     Logs Writer ( roles/logging.logWriter ) and Monitoring Metric Writer ( roles/monitoring.metricWriter ) : 
#       These allow your TPU to write logs and metrics, which are crucial for monitoring and debugging.

gcloud beta services identity create --service tpu.googleapis.com # --project DEFAULT-SEEMS-TO-APPLY-HERE

# Meh: Install (local) `gcloud alpha` package
# Meh: gcloud components install alpha
```

```bash
export TPU_NAME="kaggle-tpu-paid"
export TPU_TYPE="v5litepod-8"
export TPU_TYPE="v5litepod-1"  #  Lower cost while testing out startup scripts
export TPU_SOFTWARE="v2-alpha-tpuv5-lite"

### Broken (old?) project
##export TPU_ZONE="us-central1-f"     # DOES NOT offer me v5e-8
##export TPU_ZONE="us-central1-a"     # $1.20 per hour : NO PERMISSION
##export TPU_ZONE="us-west1-c"        # $1.20 per hour (Oregon) : NO PERMISSION
##export TPU_ZONE="us-west4-a"        # $1.20 per hour (Vegas) : NO PERMISSION
#export TPU_ZONE="asia-east1-c"       # $>?? per hour  (NO CAPACITY)
##export TPU_ZONE="asia-southeast1-b" # $1.56 per hour (but not on Where are the TPUs page) : UNKNOWN!
##export TPU_ZONE="asia-southeast1-c" # $1.56 per hour (but not on Where are the TPUs page) : UNKNOWN!

#export TPU_ZONE="us-central1-f"     # DOES NOT offer me v5e-8
#export TPU_ZONE="us-central1-a"     # $1.20 per hour : (no more capacity)
export TPU_ZONE="us-west1-c"        # $1.20 per hour (Oregon) : WORKED!
#export TPU_ZONE="us-west4-a"        # $1.20 per hour (Vegas) : (no more capacity)
#export TPU_ZONE="asia-east1-c"      # $>?? per hour  (Insufficient capacity)
#export TPU_ZONE="asia-southeast1-b" # $1.56 per hour (but not on Where are the TPUs page) : (Insufficient capacity)
#export TPU_ZONE="asia-southeast1-c" # "not supported"

```

Check the location has required TPUs:
*  eg: `gcloud compute tpus tpu-vm accelerator-types describe v5litepod-8 --zone=us-central1-a`

```bash
gcloud compute tpus accelerator-types list --zone=${TPU_ZONE}
gcloud compute tpus tpu-vm accelerator-types describe ${TPU_TYPE} --zone=${TPU_ZONE}
```



Create the TPU:

```bash
TPU_SECRET="SDFSDF"
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${TPU_ZONE} \
  --accelerator-type=${TPU_TYPE} \
  --version=${TPU_SOFTWARE} \
  --metadata-from-file=startup-script=startup_script.sh \
  --metadata=foo=${TPU_SECRET}

#  --data-disk=mode=read-write,name=YOUR_DISK_NAME
```


### Delete the TPU

```bash
gcloud compute tpus tpu-vm delete ${TPU_NAME} \
   --zone=${TPU_ZONE} \
   --quiet

#   --project=${PROJECT_ID} \    

# Check : 
gcloud compute tpus tpu-vm list --zone=${TPU_ZONE}
gcloud compute tpus tpu-vm list # Everywhere
```



#### Get the IP address:
#
#```bash
#gcloud compute config-ssh;
#export GCP_ADDR=`grep "Host ${TPU_NAME}" ~/.ssh/config | tail --bytes=+6`;
#```

### Set up port forwarding (mainly for JupyterLab / TensorBoard / ...):

```bash
#ssh ${GCP_ADDR} -L 8585:localhost:8585 -L 8586:localhost:8586 -L 5005:localhost:5005
# or
#gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_ZONE} -- -L 8585:localhost:8585
gcloud compute tpus tpu-vm ssh tpu_user@${TPU_NAME} --zone=${TPU_ZONE} -- -L 8585:localhost:8585
# Propagating SSH public key to all TPU workers...done.   
```


On the machine:

```bash

andrewsm@t1v-n-dbe8fd36-w-0:~$ python --version
Python 3.10.12

pip --version
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)

echo ${PATH}
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

.local/bin/jupyter lab --no-browser --ip=0.0.0.0 --port=$JUPYTER_PORT --allow-root
# Works!

top
# MiB Mem : 386884.3 total, 381996.2 free,   2201.3 used,   2686.8 buff/cache

df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/root        97G  9.0G   88G  10% /
tmpfs           189G     0  189G   0% /dev/shm
tmpfs            76G  2.7M   76G   1% /run
tmpfs           5.0M     0  5.0M   0% /run/lock
efivarfs         56K   24K   27K  48% /sys/firmware/efi/efivars
/dev/sda15      105M  6.1M   99M   6% /boot/efi
tmpfs            38G  4.0K   38G   1% /run/user/2001
```

### Examine Startup script output

```bash
sudo journalctl -u google-startup-scripts.service
```


### Mount drive locally?

```bash
sshfs ${GCP_ADDR}:. ./gcp_base -o reconnect -o follow_symlinks

```

Unmount drive locally?

```bash
fusermount -u ./gcp_base
```







## Attach storage?

* [attach-durable-block-storage](https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage)



## Notes

```bash
gcloud compute tpus tpu-vm accelerator-types describe v5litepod-8 --zone=asia-east1-c
<!--
acceleratorConfigs:
- topology: 2x4
  type: V5LITE_POD
name: projects/rdai-tpu-mdda/locations/asia-east1-c/acceleratorTypes/v5litepod-8
type: v5litepod-8
!-->

```


