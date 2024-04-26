################## MUJOCO
# Hopper-v2
python3 idil_train/run_algs.py alg=ogail env=Hopper-v2 base=hopper_base \
        seed=0 

# Walker2d-v2
python3 idil_train/run_algs.py alg=ogail env=Walker2d-v2 base=mujoco_base \
        seed=0 

# Humanoid-v2
python3 idil_train/run_algs.py alg=ogail env=Humanoid-v2 base=mujoco_base \
        seed=0 

# Ant-v2
python3 idil_train/run_algs.py alg=ogail env=Ant-v2 base=mujoco_base \
        seed=0 

# HalfCheetah-v2
python3 idil_train/run_algs.py alg=ogail env=HalfCheetah-v2 base=mujoco_base \
        seed=0

# AntPush-v0
python3 idil_train/run_algs.py alg=ogail env=AntPush-v0-original base=antpush_base \
        seed=0

################## Multi Goals
python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_2-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_3-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_4-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
       env=MultiGoals2D_5-v0 supervision=0.0 seed=0

################## Movers / SingleMover
python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
      env=CleanupSingle-v0 supervision=0.0 

python idil_train/run_algs.py alg=ogail base=MultiGoals2D_base \
      env=EnvMovers-v0 supervision=0.0 

