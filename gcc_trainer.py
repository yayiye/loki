import torch
from ppo import PPOTrainer
from gcc_net_env import Gccenv
import argparse
import json

episodes = 3000

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--lmbda', type=float)
    parser.add_argument('--actor_lr', type=float)
    parser.add_argument('--critic_lr', type=float)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--n_hiddens', type=int)
    args = parser.parse_args()

    torch.set_num_threads(1)
    env = Gccenv()
    trainer = PPOTrainer(env, args.gamma, args.lmbda, args.actor_lr, args.critic_lr, args.eps, args.n_hiddens)

    data = {
        "main_folder": trainer.main_folder,
        "gamma": args.gamma,
        "lmbda": args.lmbda,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "eps": args.eps,
        "n_hiddens": args.n_hiddens
    }

    print("Writing log data...")
    with open("log.json", "a+") as f:
        f.write(json.dumps(data))
        f.write("\n")
        print("Log data written successfully.")


    for i in range(episodes):
        trainer.train_one_episode()
        if i % 50 == 0:
            trainer.save()
            trainer.test_one_episode()


if __name__ == '__main__':
    main()
