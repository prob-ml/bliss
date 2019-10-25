rsync -avL --progress -e 'ssh -i ../../bryans_key_oregon.pem' \
   ubuntu@ec2-35-161-65-132.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/DeblendingStarfields/fits/.\
   /home/runjing_liu/Documents/astronomy/DeblendingStarfields/fits/.

