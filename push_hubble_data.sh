rsync -avL --progress -e 'ssh -i ./../../bryans_key_oregon.pem' \
   /home/runjing_liu/Documents/astronomy/DeblendingStarfields/hubble_data/.\
   ubuntu@ec2-34-222-117-21.us-west-2.compute.amazonaws.com:/home/ubuntu/astronomy/DeblendingStarfields/hubble_data/.

