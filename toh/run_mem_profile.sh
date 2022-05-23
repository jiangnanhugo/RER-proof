
for L in 10 20 30 40 50 60 70 80 90 100
do
    python3 trie_with_different_L.py  --L 10 --reversed --num_disks 9 --max_tryout 4096 --freeze 30 --evaluation_step 10 --play_times 1 --episodes 100 --repeats 1 > trie.L$L.log &
done


# graph
# python tabularQ_rer.py  --algo graph   --L 10 --reversed --num_disks 9 --max_tryout 4096 --freeze 30 --evaluation_step 10 --play_times 1 --episodes 100