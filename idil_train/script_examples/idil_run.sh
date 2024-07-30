################## MUJOCO
# Hopper-v2
python3 idil_train/run_algs.py alg=idil env=Hopper-v2 base=hopper_base \
        seed=0 

# Walker2d-v2
python3 idil_train/run_algs.py alg=idil env=Walker2d-v2 base=mujoco_base \
        seed=0 miql_pi_method_loss=v0

# Humanoid-v2
python3 idil_train/run_algs.py alg=idil env=Humanoid-v2 base=mujoco_base \
        seed=0 miql_pi_method_loss=v0 miql_pi_init_temp=1

# Ant-v2
python3 idil_train/run_algs.py alg=idil env=Ant-v2 base=mujoco_base \
        seed=0 miql_pi_method_loss=value miql_pi_init_temp=1e-3

# HalfCheetah-v2
python3 idil_train/run_algs.py alg=idil env=HalfCheetah-v2 base=mujoco_base \
        seed=0 miql_pi_method_loss=value

# AntPush-v0
python3 idil_train/run_algs.py alg=idil env=AntPush-v0-clipped base=antpush_base \
        seed=0 miql_pi_method_loss=value miql_pi_init_temp=1e-3

################## Multi Goals
python idil_train/run_algs.py alg=idil base=MultiGoals2D_base \
       env=MultiGoals2D_2-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=idil base=MultiGoals2D_base \
       env=MultiGoals2D_3-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=idil base=MultiGoals2D_base \
       env=MultiGoals2D_4-v0 supervision=0.0 seed=0

python idil_train/run_algs.py alg=idil base=MultiGoals2D_base \
       env=MultiGoals2D_5-v0 supervision=0.0 seed=0

################## Movers / SingleMover
python idil_train/run_algs.py alg=idil base=boxpush_base \
       env=EnvMovers-v0 seed=0 supervision=0.0

python idil_train/run_algs.py alg=idil base=boxpush_base \
       env=CleanupSingle-v0 seed=0 supervision=0.0
  

###### MISC
# Nohup with fixed expert policy (discrete)
nohup python idil_train/run_algs.py alg=idil base=boxpush_base env=CleanupSingle-v0 seed=0 supervision=0.0 fixed_pi=true  tag='disc-fep'&

# Nohup without fixed expert policy (discrete)
nohup python idil_train/run_algs.py alg=idil base=boxpush_base env=CleanupSingle-v0 seed=0 supervision=0.0 fixed_pi=false &

# Nohup with fixed expert policy (continuous)
nohup python idil_train/run_algs.py alg=idil base=MultiGoals2D_base env=MultiGoals2D_3-v0 supervision=0.0 seed=0 fixed_pi=true tag='cont-fep' &


##### ENTROPY SCORING 
# Entropy scorign @ k = 20% (Discrete env)
nohup python idil_train/run_algs.py alg=idil base=boxpush_base env=CleanupSingle-v0 seed=0 supervision=0.0 k=0.2 entropy_scoring=true tag='disc-es-20'&
# Entropy scoring @ k = 60% (Continuous env)
nohup python idil_train/run_algs.py alg=idil base=MultiGoals2D_base env=MultiGoals2D_3-v0 supervision=0.0 k=0.6 entropy_scoring=true tag='cont-es-60'&

# sweep k-values list
nohup python entropy_scoring_sweep.py --sweep_k='0.2;0.4;0.6' --num_trials=5&
# sweep k-values list (continuous env)
nohup python entropy_scoring_sweep.py --sweep_k='0.2;0.4;0.6' --base=MultiGoals2D_base --env=MultiGoals2D_3-v0 --discrete_env=false --num_trials=5 &