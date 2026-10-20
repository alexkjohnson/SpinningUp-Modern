import time
import joblib
import json
import os
import os.path as osp
import torch
import gymnasium as gym
from spinup import EpochLogger


def load_policy_and_env(fpath, itr='last', deterministic=False, render_mode=None):
    """
    Load a policy from save, along with RL env.
    """
    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        pytsave_path = osp.join(fpath, 'pyt_save')
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    env = None

    # 1. If rendering is requested, prioritize re-creating via gym.make using config.json 
    # so that render_mode='human' is correctly configured from the start.
    if render_mode is not None:
        try:
            config_path = osp.join(fpath, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                env_name = config.get('env_name')
                if env_name:
                    print(f"Re-creating environment '{env_name}' with render_mode='{render_mode}'...")
                    env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            print(f"Could not recreate environment from config: {e}")

    # 2. Fallback to loading environment from pickle if config creation failed or render_mode is None
    if env is None:
        try:
            state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
            env = state['env']
        except:
            pass

    # 3. Last resort fallback if env still has no render_mode but we need one
    if env is not None and hasattr(env, 'render_mode') and env.render_mode is None and render_mode is not None:
        try:
            env_id = env.spec.id if env.spec else None
            if env_id:
                env = gym.make(env_id, render_mode=render_mode)
        except:
            pass

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    # weights_only=False required for PyTorch 2.6+ when loading full model classes
    model = torch.load(fname, weights_only=False)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset()[0], 0, False, 0, 0, 0
    while n < num_episodes:
        if render and hasattr(env, 'render_mode') and env.render_mode == 'human':
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset()[0], 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    render_mode = None if args.norender else 'human'

    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic,
                                          render_mode=render_mode)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))