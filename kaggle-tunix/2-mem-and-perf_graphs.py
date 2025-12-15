# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


# -

def regplot(df, x="steps_max", y="elapsed", include_origin=True):
  fig, ax = plt.subplots(figsize=(12,4))
  sns.regplot(data=df, x=x, y=y, ax=ax, truncate=False )
  if include_origin:
    ax.set(xlim=(0, None), ylim=(0, None))  # Choose defaults for max
  plt.show()


import pickle
def load_and_annotate(filename, annotations=dict()):
  with open(filename, 'rb') as f:
    data = pickle.load(f)  
    for d in data:
      for k,v in annotations.items():
        d[k]=v
      if d.get('lora', False) and d.get('qwix', False):  
        d['hbm0'] -= 2000512000   # qwix version has two models loaded
      d['hbm0'] /= 1000*1000*1000 # i.e. in decimal GB
      d['ms_per_token'] = d['elapsed']*1000. / (d['steps_max']*d['batch_size'])
  return data


# ### Look at scaling vs number of steps
#
# * Appears to be entirely linear...

res_df = pd.DataFrame(
  # different numbers of steps, batch_size=1
  load_and_annotate('docs/steps-vary_bs1.pkl'), 
)
regplot(res_df, y="elapsed")  # When steps varied

regplot(res_df, x="steps_max", y="hbm0")

# ### Look at trends for specific steps across batch_size
#
# * Detail view shows that `batch_size==64` and `batch_size==128`
#   + much better than `batch_size==68` and `batch_size==132`

res_df = pd.DataFrame(
  # steps=1024, lots of batch_sizes
  #load_and_annotate('docs/bs-to-200_steps1024.pkl', dict(lora=False)), 
  #load_and_annotate('docs/bs-to-128_steps1024_lora.pkl', dict(lora=True, qwix=True)), 
  #load_and_annotate('docs/bs-detail-128_steps1024_lora.pkl', dict(lora=True, qwix=True)), 
  #load_and_annotate('docs/bs-more-detail-128_steps1024_lora.pkl', dict(lora=True, qwix_fixed=True)), 
  load_and_annotate('docs/logits_bs_steps1024_lora.pkl', dict(lora=True, qwix_fixed=True)), 
)

regplot(res_df, x="batch_size", y="elapsed")

regplot(res_df, x="batch_size", y="ms_per_token")

regplot(res_df, x="batch_size", y="hbm0")

# ### Now do comparison of basic vs LoRA speeds
#
# * Shows that LoRA hardly affects token generation speed
#   + Compared to multiple of 64 `batch_size` choice...

# +
basic_df = pd.DataFrame(
  load_and_annotate('docs/bs-to-200_steps1024.pkl', dict(lora=False)), 
)

lora1_df = pd.DataFrame(
  load_and_annotate('docs/bs-to-128_steps1024_lora.pkl', dict(lora=True, qwix=True)), 
)
lora2_df = pd.DataFrame(
#  load_and_annotate('docs/bs-detail-128_steps1024_lora.pkl', dict(lora=True, qwix=True)), 
  load_and_annotate('docs/bs-more-detail-128_steps1024_lora.pkl', dict(lora=True, qwix_fixed=True)), 
)
combo_df =  pd.concat([basic_df, lora1_df, lora2_df])

# +
#combo_df[:4]
#combo_df[combo_df['batch_size']>6]
#combo_df[combo_df['lora']]

# +
bs_min, bs_max = 30, 150

fig, ax = plt.subplots(figsize=(12,4))
sns.regplot(data=combo_df[(combo_df['batch_size']>bs_min) & (combo_df['batch_size']<bs_max)& ~combo_df['lora']], 
            x="batch_size", y="ms_per_token", ax=ax, truncate=False,
           line_kws={'label': 'Basic'})
sns.regplot(data=combo_df[(combo_df['batch_size']>bs_min) & (combo_df['batch_size']<bs_max)& combo_df['lora']], 
            x="batch_size", y="ms_per_token", ax=ax, truncate=False,
           line_kws={'label': 'LoRA'})
ax.legend(loc="best")
plt.show()
# -




