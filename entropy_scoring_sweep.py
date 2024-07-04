import subprocess
import argparse

COMMAND_STR_TEMPLATE = '''
python idil_train/run_algs.py alg={alg} base={base} env={env} seed=0 supervision=0.0 k={kval} entropy_scoring=true tag='{env_type}-es-{kval_label}'
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_k", type=str, default="0.2;0.5", required=True,
                        help="Sweep values for entropy top-K selection, separated by ';'")
    parser.add_argument("--discrete_env", type=bool, default=True, required=False,
                        help="Whether to use discrete environment")
    parser.add_argument("--alg", type=str, default="idil", required=False,
                        help="Algorithm to run")
    parser.add_argument("--base", type=str, default="boxpush_base", required=False,
                        help="Base directory for the experiment")
    parser.add_argument("--env", type=str, default="CleanupSingle-v0", required=False,
                        help="Environment to run the algorithm on")
    
    args = parser.parse_args()
    sweep_k_list = args.sweep_k.split(";")

    env_type = "disc" if args.discrete_env else "cont"

    print(f"Gotten sweep Ks: {sweep_k_list}")

    for sweep_k in sweep_k_list:
        k_label = str(int(float(sweep_k) * 100))
        command_str = COMMAND_STR_TEMPLATE.format(alg=args.alg,
                                                  base=args.base,
                                                  env=args.env, 
                                                  env_type=env_type,
                                                  kval=sweep_k, 
                                                  kval_label=k_label)
        
        print(f"\n\nRunning command: {command_str}\n\n")
        subprocess.run(command_str, shell=True)
