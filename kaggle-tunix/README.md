## kaggle-tunix notes

### GCP set-up

* [TPU pricing](https://cloud.google.com/tpu/pricing)
* [Where are the TPUs?](https://docs.cloud.google.com/tpu/docs/regions-zones)
* [TPU Software versions](https://docs.cloud.google.com/tpu/docs/runtimes)

```bash
# gcloud auth login
# gcloud services enable tpu.googleapis.com

# Grant Essential IAM Roles to <YOUR_GCP_USER> :
#   Your account needs specific permissions to create and manage TPUs. The most critical role is TPU Admin .
#     TPU Admin ( roles/tpu.admin ): This role grants comprehensive access to Cloud TPU resources, 
#     including the `tpu.nodes.create` permission, which is essential for creating new TPUs.
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
export TPU_SOFTWARE="v2-alpha-tpuv5-lite"

#export TPU_ZONE="us-central1-a"     # $1.20 per hour : NO PERMISSION
#export TPU_ZONE="us-west1-c"        # $1.20 per hour (Oregon) : NO PERMISSION
#export TPU_ZONE="us-west4-a"        # $1.20 per hour (Vegas) : NO PERMISSION
export TPU_ZONE="asia-east1-c"       # $>?? per hour  (NO CAPACITY)
#export TPU_ZONE="asia-southeast1-b" # $1.56 per hour (but not on Where are the TPUs page) : UNKNOWN!
#export TPU_ZONE="asia-southeast1-c" # $1.56 per hour (but not on Where are the TPUs page) : UNKNOWN!

```

Create the TPU:

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${TPU_ZONE} \
  --accelerator-type=${TPU_TYPE} \
  --version=${TPU_SOFTWARE}
  
#   \
#  --metadata-from-file=startup-script=startup_script.sh  
```



Get the IP address:

```bash
gcloud compute config-ssh;
export GCP_ADDR=`grep "Host ${TPU_NAME}" ~/.ssh/config | tail --bytes=+6`;
```

Set up port forwarding (mainly for JupyterLab / TensorBoard / ...):

```bash
#ssh ${GCP_ADDR} -L 8585:localhost:8585 -L 8586:localhost:8586 -L 5005:localhost:5005
# or
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_ZONE} -- -L 8585:localhost:8585
```


Mount drive locally?

```bash
sshfs ${GCP_ADDR}:. ./gcp_base -o reconnect -o follow_symlinks

```

Unmount drive locally?

```bash
fusermount -u ./gcp_base
```



## Attach storage?

* [attach-durable-block-storage](https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage)



