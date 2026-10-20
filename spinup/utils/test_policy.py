import time
import joblib
import json
import os
import os.path as osp
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from spinup import EpochLogger
import os
# Force MuJoCo to use a robust offscreen rendering backend for video recording
os.environ["MUJOCO_GL"] = "egl"  # or "osmesa" if egl is unavailable


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
    env_name = None

    # 1. Try reading env name from config.json ('env' is standard in Spinning Up)
    try:
        config_path = osp.join(fpath, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            env_val = config.get('env') or config.get('env_name')
            if isinstance(env_val, str):
                env_name = env_val
    except Exception as e:
        print(f"Could not read config.json: {e}")

    # 2. If config didn't give a string name, inspect the pickled vars file
    if env_name is None:
        try:
            state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
            raw_env = state.get('env')
            if raw_env is not None:
                if hasattr(raw_env, 'spec') and raw_env.spec is not None and raw_env.spec.id:
                    env_name = raw_env.spec.id
                elif hasattr(raw_env, 'unwrapped') and hasattr(raw_env.unwrapped, 'spec') and raw_env.unwrapped.spec is not None:
                    env_name = raw_env.unwrapped.spec.id
                else:
                    # Fallback based on class name inspection
                    class_name = raw_env.__class__.__name__
                    if 'HalfCheetah' in class_name:
                        env_name = 'HalfCheetah-v5'
                    else:
                        env_name = 'HalfCheetah-v5'  # Default fallback for your Mujoco experiments
        except Exception as e:
            pass

    # 3. Instantiate the environment via gym.make with the requested render_mode
    if env_name is not None:
        try:
            print(f"Re-creating environment '{env_name}' with render_mode='{render_mode}'...")
            env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            print(f"Could not make environment '{env_name}': {e}")

    # 4. Final fallback if gym.make failed
    if env is None:
        try:
            state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
            env = state['env']
            if render_mode is not None and hasattr(env, 'render_mode'):
                env.render_mode = render_mode
        except Exception as e:
            print(f"Could not load environment from pickle: {e}")

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

    # Ensure environment (and video writer wrapper) closes and flushes to disk
    if env is not None:
        env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--record', '-r', action='store_true', help='Record video of test episodes')
    parser.add_argument('--video-dir', type=str, default='videos', help='Directory to save recorded videos')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    # Determine render mode based on flags
    if args.record:
        render_mode = 'rgb_array'
    elif args.norender:
        render_mode = None
    else:
        render_mode = 'human'

    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic,
                                          render_mode=render_mode)

    # Apply video recording wrapper if requested
    if args.record and env is not None:
        print(f"Recording videos to '{args.video_dir}' folder...")
        env = RecordVideo(env, video_folder=args.video_dir, episode_trigger=lambda episode_id: episode_id < args.episodes)

    run_policy(env, get_action, args.len, args.episodes, render=(not args.norender and not args.record))