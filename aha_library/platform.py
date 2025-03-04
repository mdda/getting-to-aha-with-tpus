import os, subprocess

def detect(XLA_PYTHON_CLIENT_MEM_FRACTION=1.0):
  try:
    subprocess.check_output('nvidia-smi')
    return 'gpu'
  except:
    # We're not on a cuda machine - let's see whether we're on a TPU one
    if 'TPU_ACCELERATOR_TYPE' in os.environ:
      return 'tpu'
    else:  # We are on a CPU machine
      return 'cpu'        

def jax_pip_install_str(backend, XLA_PYTHON_CLIENT_MEM_FRACTION=1.0):
  s, accelerator = '', True
  if backend=='cpu':
    s='pip install -U "jax"'
    accelerator = False
  elif backend=='gpu':
    s='pip install -U "jax[cuda12]"'
  elif backend=='tpu':
    s='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'
  else:
    raise(f"Unknown backend '{backend}'")
  if accelerator:
    # By default JAX will preallocate 75% of the total GPU memory when the first JAX operation is run. 
    #   https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(XLA_PYTHON_CLIENT_MEM_FRACTION)
  return s

