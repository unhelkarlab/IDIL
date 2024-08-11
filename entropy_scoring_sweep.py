import subprocess
import argparse
from ulid import ULID
import numpy as np

COMMAND_STR_TEMPLATE = '''
python idil_train/run_algs.py alg={alg} base={base} env={env} seed={seed} supervision={supervision} k={kval} entropy_scoring=true tag='{tag}' fixed_pi={fixed_pi}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_k", type=str, default="0.2;0.5", required=True,
                        help="Sweep values for entropy top-K selection, separated by ';'")
    parser.add_argument("--num_trials", type=int, default=1, required=False,
                        help="Number of trials to run for each K value")
    parser.add_argument("--supervision", type=float, default=0.5, required=False,
                        help="Proportion of labels to be used in the expert dataset")
    parser.add_argument("--discrete_env", type=bool, default=True, required=False,
                        help="Whether to use discrete environment")
    parser.add_argument("--alg", type=str, default="idil", required=False,
                        help="Algorithm to run")
    parser.add_argument("--base", type=str, default="boxpush_base", required=False,
                        help="Base directory for the experiment")
    parser.add_argument("--env", type=str, default="CleanupSingle-v0", required=False,
                        help="Environment to run the algorithm on")
    parser.add_argument("--custom_tag", type=str, required=False,
                        help="Custom tag for the experiment, appended to the standard ID")
    parser.add_argument("--fixed_pi", type=bool, default=False, required=False,
                        help="Whether to use a fixed action policy (expert)")
    
    args = parser.parse_args()
    sweep_k_list = args.sweep_k.split(";")

    env_type = "disc" if args.discrete_env else "cont"

    print(f"Gotten sweep Ks: {sweep_k_list}")

    assert isinstance(args.fixed_pi, bool), "fixed_pi must be a boolean"
    
    for sweep_k in sweep_k_list:
        for trial_id in range(args.num_trials):
            job_id = ULID()

            # assign a random seed for each job
            seed = np.random.randint(1, 100000)
            
            k_label = str(int(float(sweep_k) * 100))

            # create job tag
            
            tag = f'es-{k_label}-{job_id}'
            if args.custom_tag:
                tag = f'{tag}-{args.custom_tag}'

            command_str = COMMAND_STR_TEMPLATE.format(alg=args.alg,
                                                    base=args.base,
                                                    env=args.env,
                                                    seed=seed, 
                                                    supervision=args.supervision,
                                                    env_type=env_type,
                                                    kval=sweep_k, 
                                                    tag=tag,
                                                    fixed_pi=args.fixed_pi)
            
            print(f"\n\nRunning command: {command_str}\n\n")
            subprocess.run(command_str, shell=True)
