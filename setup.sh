conda create -n nerfacc python=3.9 -y
conda activate nerfacc
conda install pytorch cudatoolkit=11.3 -c pytorch -y
pip install -e .