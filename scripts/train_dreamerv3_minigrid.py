import warnings
import functools
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import gymnasium as gym
import ruamel.yaml as yaml

# 1. Setup Paths & Mocking
# Add third_party to path so we can import dreamerv3 and embodied
sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/dreamerv3"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Mock 'gym' with 'gymnasium' to satisfy embodied imports
sys.modules["gym"] = gym

try:
    import dreamerv3
    import embodied
    import elements
except ImportError as e:
    print(f"Error: {e}")
    print("Please follow the installation instructions in README.md.")
    sys.exit(1)

import check_dreamer_env_io
import lifelong_learning.envs # Register envs

# 2. Custom Adapter for Gymnasium
class FromGymnasium(embodied.Env):
    """
    Adapter to wrap Gymnasium environments for Embodied agents.
    Handles the 5-tuple step return (obs, reward, terminated, truncated, info).
    """
    def __init__(self, env, obs_key='image', act_key='action', **kwargs):
        if isinstance(env, str):
            self._env = gym.make(env, **kwargs)
        else:
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces['reset'] = elements.Space(bool)
        return spaces

    def step(self, action):
        if self._done or (isinstance(action, dict) and action.get('reset', False)):
            self._done = False
            obs, info = self._env.reset()
            self._info = info
            return self._obs(obs, 0.0, is_first=True)
            
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
            
        # Gymnasium Step: 5 values
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        self._done = terminated or truncated
        self._info = info
        return self._obs(
            obs, reward,
            is_last=bool(self._done),
            is_terminal=bool(info.get('is_terminal', terminated)) # Dreamer uses is_terminal for True termination
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal
        )
        return obs

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + '/' + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split('/')
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, 'n'):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)


def main():
    # 3. Define Config
    # Load configs from dreamerv3 directory
    dv3_path = os.path.dirname(dreamerv3.__file__)
    config_path = os.path.join(dv3_path, 'configs.yaml')
    
    # Use modern ruamel.yaml API
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open(config_path, 'r') as f:
        configs = yaml_loader.load(f)
        
    # Inject custom env settings into defaults so elements.Flags can parse them
    configs['defaults']['env']['steps_per_regime'] = 0
    configs['defaults']['env']['episodes_per_regime'] = 0
    configs['defaults']['env']['oracle_mode'] = False
    configs['defaults']['env']['id'] = "MiniGrid-DualGoal-8x8-v0"

    config = elements.Config(configs['defaults'])
    
    # Handle --configs flag (e.g. --configs size1m)
    # Parse it before other flags, just like DreamerV3's main.py
    parsed, remaining_argv = elements.Flags(configs=['size12m']).parse_known()
    for name in parsed.configs:
        config = config.update(configs[name])
    
    config = config.update({
        'logdir': f'./logdir/{datetime.now().strftime("%Y%m%d-%H%M%S")}-dualgoal',
        'run.train_ratio': 32,
        'run.log_every': 30,
        'batch_size': 16,
        'replay_context': 0,
    })

    # Check I/O if requested
    check_io = False
    if '--check_env_io' in sys.argv:
        try:
            idx = sys.argv.index('--check_env_io')
            if idx + 1 < len(sys.argv):
                val = sys.argv[idx+1]
                check_io = (val.lower() == 'true')
            else:
                check_io = True
        except:
             check_io = True

    if check_io:
        print("Running I/O Sanity Check...")
        check_dreamer_env_io.check_env_io(
            env_id=config.env.id, 
            oracle=config.env.oracle_mode
        )
        print("Sanity Check Passed. Exiting (remove --check_env_io to train).")
        return

    config = elements.Flags(config).parse(remaining_argv)
    
    bind = functools.partial

    print('Logdir:', config.logdir)
    print('Env ID:', config.env.id)
    print('Regime Steps:', config.env.steps_per_regime)

    # 4. Helper Functions using FromGymnasium
    
    def make_env(config, index, **overrides):
        from lifelong_learning.envs.make_env import make_env as ll_make_env
        from dreamerv3 import main as dv3_main
        
        steps = config.env.steps_per_regime
        if steps == 0: steps = None

        episodes = config.env.episodes_per_regime
        if episodes == 0: episodes = None
        
        # Convert global steps/episodes to per-env counts.
        # Each parallel env has its own counter, so divide by num_envs
        # to make the CLI value correspond to global training steps.
        num_envs = config.run.envs
        if steps is not None:
            steps = max(1, int(steps) // num_envs)
        if episodes is not None:
            episodes = max(1, int(episodes) // num_envs)
        
        # Calculate seed based on index
        # index is passed by embodied.run.train for each parallel env
        seed = config.seed + index
        
        env = ll_make_env(
            env_id=config.env.id,
            seed=seed,
            dreamer_compatible=True,
            oracle_mode=config.env.oracle_mode,
            episodes_per_regime=episodes,
            steps_per_regime=steps,
        )
        # Use our custom wrapper instead of from_gym
        env = FromGymnasium(env, obs_key='image', act_key='action')
        env = dv3_main.wrap_env(env, config)
        return env

    def make_agent(config):
        from dreamerv3 import agent as dv3_agent
        # Create a dummy env to get spaces
        env = make_env(config, 0)
        
        # Agent expects a specific config structure (flat agent config + some globals)
        # See dreamerv3/main.py make_agent
        agent_config = elements.Config(
            **config.agent,
            logdir=config.logdir,
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        )

        # IMPORTANT: Agent must NOT see reset or log/ keys
        # log/ keys are for TensorBoard logging only, not neural net inputs
        notlog = lambda k: not k.startswith('log/')
        agent_obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
        agent_act_space = dict(env.act_space)
        agent_act_space.pop('reset', None)

        agent = dv3_agent.Agent(agent_obs_space, agent_act_space, agent_config)
        env.close()
        return agent

    def make_logger(config):
        logdir = elements.Path(config.logdir)
        step = elements.Counter()
        return elements.Logger(step, [
            elements.logger.TerminalOutput(),
            elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
            elements.logger.TensorBoardOutput(logdir),
        ])

    def make_replay(config, folder, mode='train'):
        # Match upstream length calculation: consec * batlen + context
        batlen = config.batch_length if mode == 'train' else config.report_length
        consec = config.consec_train if mode == 'train' else config.consec_report
        length = consec * batlen + config.replay_context
        return embodied.Replay(
            length=length,
            capacity=int(config.replay['size']),
            directory=elements.Path(config.logdir) / folder,
        )

    def make_stream(config, replay, mode):
        fn = bind(replay.sample, config.batch_size, mode)
        stream = embodied.streams.Stateless(fn)
        stream = embodied.streams.Consec(
            stream,
            length=config.batch_length if mode == 'train' else config.report_length,
            consec=config.consec_train if mode == 'train' else config.consec_report,
            prefix=config.replay_context,
            strict=(mode == 'train'),
            contiguous=True)
        return stream

    # 5. Start Training
    args = elements.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )
    
    # embodied.run.train expects factories, not objects
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args
    )

if __name__ == "__main__":
    main()
