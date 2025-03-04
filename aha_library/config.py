import os
from omegaconf import OmegaConf

def read(BASE='./'):
  config = OmegaConf.load(f'{BASE}/config.yaml')
  for extra in [f'{BASE}/config_secrets.yaml']:
    if os.path.isfile(extra):
      config = OmegaConf.merge(config, OmegaConf.load(extra))
  return config
  
def load_kaggle_secrets(config):
  got_kaggle=False
  try:
    from google.colab import userdata
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')    
    got_kaggle=True
  except:
    # We're not on Colab.., or had no permissions
    pass
    
  if not got_kaggle:
    try:
      # Read in from the configuration files
      os.environ['KAGGLE_USERNAME'] = config.kaggle.username
      os.environ['KAGGLE_KEY'] = config.kaggle.key
      got_kaggle=True
    except:
      print(f"Need to have kaggle secrets in {BASE}/config_secrets.yaml")
    
  return got_kaggle
