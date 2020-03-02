#!/bin/sh
echo ~~~ starting docker... ~~~
sudo service docker start

if [ ! "$(sudo docker ps -a | grep gen-pset1)" ]; then
        echo ~~~ booting container for the first time... ~~~
        sudo docker run \
            -it \
            --name gen-pset1 \
            --publish 8080:8080/tcp \
            --publish 8090:8090/tcp \
            --publish 8091:8091/tcp \
            --publish 8092:8092/tcp \
            probcomp/mit-6.885-spring2020-gen-student-pset1
else
        sudo docker start -i gen-pset1
fi
