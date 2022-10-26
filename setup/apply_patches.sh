cat ../setup/analyzer.patch | patch -R /root/miniconda3/envs/torchbeast/lib/python3.9/site-packages/neurips2022nmmo/evaluation/analyzer.py
cat ../setup/metrics.patch | patch -R /root/miniconda3/envs/torchbeast/lib/python3.9/site-packages/neurips2022nmmo/env/metrics.py
cat ../setup/tasks.patch | patch -R /root/miniconda3/envs/torchbeast/lib/python3.9/site-packages/neurips2022nmmo/tasks.py 
