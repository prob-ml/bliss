rsync -avL --progress -e 'ssh -i ./../../bryans_key_oregon.pem' \
   /home/runjing_liu/Documents/astronomy/DeblendingStarfields/hubble_data/.\
   ubuntu@ec2-34-219-182-18.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/DeblendingStarfields/hubble_data/.

