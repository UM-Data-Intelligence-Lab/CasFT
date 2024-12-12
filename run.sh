# twitter -1d
python train.py --data=twitter --observation_time=86400 --prediction_time=1296000 --max_events=10000 --interval_num=12 --tpp_hdims=64 --FC=1 --lr=2e-3 --timesteps=1000
# twitter -2d
python train.py --data=twitter --observation_time=172800 --prediction_time=1296000 --max_events=15000 --interval_num=32 --tpp_hdims=128 --FC=1 --lr=2e-3 --timesteps=1000

# aps -3y
python train.py --data=aps --observation_time=1095 --prediction_time=7305 --max_events=10000 --interval_num=20 --tpp_hdims=64 --FC=0 --lr=2e-3 --timesteps=1000
# aps -5y
python train.py --data=aps --observation_time=1826 --prediction_time=7305 --max_events=10000 --interval_num=8 --tpp_hdims=64 --FC=0 --lr=2e-3 --timesteps=1000

# weibo -0.5h
python train.py --data=weibo --observation_time=1800 --prediction_time=86400 --max_events=10000 --interval_num=32 --tpp_hdims=128 --FC=1 --lr=2e-3 --timesteps=1500
# weibo -1h
python train.py --data=weibo --observation_time=3600 --prediction_time=86400 --max_events=15000 --interval_num=4 --tpp_hdims=128 --FC=1 --lr=2e-3 --timesteps=1500