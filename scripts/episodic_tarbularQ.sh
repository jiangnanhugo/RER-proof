
for chunksize in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150
do
  python3 gym_control/Epi-tabularQ_treeER.py  --L $chunksize --reversed --num_disks 7 --max_tryout 4096 --freeze 10 --evaluation_step 5 --play_times 10
done
