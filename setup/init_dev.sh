git clone https://github.com/daveey/platypus-nmmo-nips.git
echo 'alias de="sudo docker exec -ti platypus /bin/bash"' >> ~/.bashrc
echo 'alias db="sudo docker build -t platypus ."' >> ~/.bashrc
echo 'alias dr="bash ~/platypus-nmmo-nips/setup/docker_run.sh"' >> ~/.bashrc
echo 'alias ds="sudo docker stop platypus; sudo docker rm platypus"' >> ~/.bashrc
