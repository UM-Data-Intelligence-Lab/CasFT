#twitter
(
python gen_cas.py --data=data/twitter/ --observation_time=86400 --prediction_time=1296000 --interval_num=12
python gen_emb.py --data=data/twitter/ --observation_time=86400 --max_seq=100
) &
(
python gen_cas.py --data=data/twitter/ --observation_time=172800 --prediction_time=1296000 --interval_num=32
python gen_emb.py --data=data/twitter/ --observation_time=172800 --max_seq=100
) &

#weibo
(
python gen_cas.py --data=data/weibo/ --observation_time=1800 --prediction_time=86400 --interval_num=4
python gen_emb.py --data=data/weibo/ --observation_time=1800 --max_seq=100
) &

(
python gen_cas.py --data=data/weibo/ --observation_time=3600 --prediction_time=86400 --interval_num=32
python gen_emb.py --data=data/weibo/ --observation_time=3600 --max_seq=100
) &



#aps
(
python gen_cas.py --data=data/aps/ --observation_time=1095 --prediction_time=7305 --interval_num=20
python gen_emb.py --data=data/aps/ --observation_time=1095 --max_seq=100
) &
(
python gen_cas.py --data=data/aps/ --observation_time=1826 --prediction_time=7305 --interval_num=8
python gen_emb.py --data=data/aps/ --observation_time=1826 --max_seq=100
) &