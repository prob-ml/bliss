# my chain is saved in a public google drive. 
# here is a script to download the chain. 
# adapted from \\https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/#WGET_Command_For_Large_over_100MB_Files

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1y6QxxiG6akgPDHVwoGpbf4TYIdMZgyBX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1y6QxxiG6akgPDHVwoGpbf4TYIdMZgyBX" -O my_chain_nsamp3000.npz && rm -rf /tmp/cookies.txt

