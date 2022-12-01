# CAMELS Graphwavenet
Code repo for graphwavenet on CAMELS

These are most of the files used in my CAMELS WRR work (The CAMELS data loaders were too large to be uploaded).

The original WRR paper is 

*Sun, Alexander Y., Peishi Jiang, Maruti K. Mudunuru, and Xingyuan Chen. "Explore Spatio‐Temporal Learning of Large Sample Hydrology Using Graph Neural Networks." Water Resources Research 57, no. 12 (2021): e2021WR030394.*

The original GraphWaveNet paper is 

*Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph wavenet for deep spatial-temporal graph modeling. arXiv preprint arXiv:1906.00121.*


To use for your own work, just implement the data loaders as requested.

- `readcamels.py`, this is the main data preparation file for using CAMELS data
- `trainwavenetwu.py`, this is the main driver of GraphWaveNet
- `gwnetmodel.py`, the original GraphWaveNet model by Wu et al.
- `util_gtnet.py`, contains utility functions
- `utils_wnet.py`, contains utility functions

To reproduce my paper results, run `runsample`


