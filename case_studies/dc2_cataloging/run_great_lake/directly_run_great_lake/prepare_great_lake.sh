echo "copy cached data from nfs to /scratch"
mkdir -p /scratch/regier_root/regier0/pduan/dc2local/
rsync -av --info=progress2 \
 /nfs/turbo/lsa-regier/scratch/pduan/dc2_cached_data \
 /scratch/regier_root/regier0/pduan/dc2local/
mkdir -p ~/bliss_output/
ln -sf /scratch/regier_root/regier0/pduan/dc2local/dc2_cached_data ~/bliss_output/dc2_cached_data
