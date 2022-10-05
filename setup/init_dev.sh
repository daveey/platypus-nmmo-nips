git clone https://github.com/daveey/platypus-nmmo-nips.git
echo 'alias de="sudo docker exec -ti platypus /bin/bash"' >> ~/.bashrc
echo 'alias db="sudo docker build -t platypus ."' >> ~/.bashrc
echo 'alias dr="bash ~/platypus-nmmo-nips/setup/docker_run.sh"' >> ~/.bashrc
echo 'alias dt="bash ~/platypus-nmmo-nips/setup/docker_train.sh"' >> ~/.bashrc
echo 'alias dtc="bash ~/platypus-nmmo-nips/setup/docker_train_clean.sh"' >> ~/.bashrc
echo 'alias ds="sudo docker stop platypus; sudo docker rm platypus"' >> ~/.bashrc

source ~/.bashrc
cd platypus-nmmo-nips


sudo docker run -d --name platypus  -v /mnt/shared:/mnt/shared --gpus 1 --shm-size=40G platypus /bin/bash -c "cd src/baseline; bash train.sh"
de -c "python plot.py; mv plot.png /mnt/shared/"


 de -c "ls -t results/nmmo | grep model | head -n 5" 
 de -c "python eval.py --model results/nmmo/model_96512.pt"
