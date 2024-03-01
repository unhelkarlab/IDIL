################## MUJOCO
# Hopper-v2
python3 train_dnn/run_algs.py alg=iql env=Hopper-v2 base=hopper_base \
        tag=oct3 seed=0 method_loss=v0 init_temp=1e-2

# Walker2d-v2
python3 train_dnn/run_algs.py alg=iql env=Walker2d-v2 base=mujoco_base \
        tag=oct3 seed=0 method_loss=v0 init_temp=1e-2

# Humanoid-v2
python3 train_dnn/run_algs.py alg=iql env=Humanoid-v2 base=mujoco_base \
        tag=oct3 seed=0 method_loss=v0 init_temp=0.3

# Ant-v2
python3 train_dnn/run_algs.py alg=iql env=Ant-v2 base=mujoco_base \
        tag=oct3 seed=0 method_loss=value init_temp=1e-3

# HalfCheetah-v2
python3 train_dnn/run_algs.py alg=iql env=HalfCheetah-v2 base=mujoco_base \
        tag=oct3 seed=0 method_loss=value init_temp=1e-2

# AntPush-v0
python3 train_dnn/run_algs.py alg=iql env=AntPush-v0-clipped base=antpush_base \
        tag=oct3 seed=0 method_loss=value init_temp=1e-3

################## Multi Goals
python train_dnn/run_algs.py alg=iql base=MultiGoals2D_base \
       env=MultiGoals2D_2-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=iql base=MultiGoals2D_base \
       env=MultiGoals2D_3-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=iql base=MultiGoals2D_base \
       env=MultiGoals2D_4-v0 tag=Sv0 supervision=0.0 seed=0

python train_dnn/run_algs.py alg=iql base=MultiGoals2D_base \
       env=MultiGoals2D_5-v0 tag=Sv0 supervision=0.0 seed=0

################## Movers / SingleMover
python train_dnn/run_algs.py alg=iql base=boxpush_base \
       env=CleanupSingle-v0 tag=sv0 seed=0 supervision=0.0 

python train_dnn/run_algs.py alg=iql base=boxpush_base \
       env=EnvMovers-v0 tag=sv0 seed=0 supervision=0.0 
