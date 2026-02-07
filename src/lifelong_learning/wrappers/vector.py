
import collections
import gymnasium as gym
import numpy as np

class VectorRollingAvgWrapper(gym.vector.VectorWrapper):
    """
    Vector implementation of RollingAvgWrapper.
    Requires RecordEpisodeStatistics to be applied before this wrapper.
    """
    def __init__(self, env, window_size=50):
        super().__init__(env)
        self.return_queues = [collections.deque(maxlen=window_size) for _ in range(self.num_envs)]
        self.length_queues = [collections.deque(maxlen=window_size) for _ in range(self.num_envs)]
        
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Check for finished episodes using _episode mask (standard in Gymnasium Vector)
        if "_episode" in infos:
             # Iterate over indices where _episode is True
             for i in np.where(infos["_episode"])[0]:
                 # Get return/length from info['episode'] (added by RecordEpisodeStatistics)
                 if "episode" in infos:
                     # episode info is usually a dict of arrays in VectorEnv
                     ret = infos["episode"]["r"][i]
                     length = infos["episode"]["l"][i]
                     
                     self.return_queues[i].append(ret)
                     self.length_queues[i].append(length)
                     
                     # Add avg to info
                     if "episode_avg" not in infos:
                          infos["episode_avg"] = {
                              "r": np.zeros(self.num_envs, dtype=np.float32), 
                              "l": np.zeros(self.num_envs, dtype=np.float32)
                          }
                     
                     # Fill values
                     infos["episode_avg"]["r"][i] = np.mean(self.return_queues[i])
                     infos["episode_avg"]["l"][i] = np.mean(self.length_queues[i])
                         
        return obs, rewards, terminations, truncations, infos
