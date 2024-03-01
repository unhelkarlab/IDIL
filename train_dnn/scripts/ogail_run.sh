################## MUJOCO
# Hopper-v2
python3 train_dnn/run_algs.py alg=ogail env=Hopper-v2 base=hopper_base \
        tag=oct3 seed=0 

# Walker2d-v2
python3 train_dnn/run_algs.py alg=ogail env=Walker2d-v2 base=mujoco_base \
        tag=oct3 seed=0 

# Humanoid-v2
python3 train_dnn/run_algs.py alg=ogail env=Humanoid-v2 base=mujoco_base \
        tag=oct3 seed=0 

# Ant-v2
python3 train_dnn/run_algs.py alg=ogail env=Ant-v2 base=mujoco_base \
        tag=oct3 seed=0 

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=ogail env=HalfCheetah-v2 base=mujoco_base \
        tag=oct3 seed=0

# AntPush-v0
python3 train_dnn/run_algs.py alg=ogail env=AntPush-v0-original base=antpush_base \
        tag=oct3 seed=0

################## Multi Goals
python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_2-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_3-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_4-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_5-v0 tag=Sv0 supervision=0.0 seed=0

################## Movers / SingleMover
python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
      env=CleanupSingle-v0 tag=sv0 supervision=0.0 

python train_dnn/run_algs.py alg=ogail base=MultiGoals2D_base \
      env=EnvMovers-v0 tag=sv0 supervision=0.0 

