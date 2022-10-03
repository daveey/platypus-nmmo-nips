import unittest
import json
from neural_mmo.train_wrapper import TrainEnv

class MockTrainEnv(TrainEnv):
    def __init__(self, *args, **kwargs):
        self._num_team_members = 8
        self._dummy_body_feature = {}
        pass

class TestTrainWrapper(unittest.TestCase):
    def test_obs_team_to_mb_agent(self):
        env = MockTrainEnv()

        team_obs =  {
                        1: {
                            3: {
                                "Entity": { "Continuous": [[3,3,3,3,3]], "N": [8] },
                                "Item": { "Continuous": [[3,0,0,0,0]], "N": [1] },
                                "Market": { "Continuous": [[3,0,0,0,0]], "N": [0] },
                                "Tile": { "Continuous": [[0,0,3,3]], "N": [15] }
                            },
                            4: {
                                "Entity": { "Continuous": [[4,4,4,4,4]], "N": [8] }, 
                                "Item": { "Continuous": [[4,0,0,0,0]], "N": [1] }, 
                                "Market": { "Continuous": [[4,0,0,0,0]], "N": [0] }, 
                                "Tile": { "Continuous": [[0,0,4,4]], "N": [15] }
                            }
                        },
                        2: {
                            5: {
                                "Entity": { "Continuous": [[5,5,5,5,5]], "N": [8] },
                                "Item": { "Continuous": [[5,0,0,0,0]], "N": [1] },
                                "Market": { "Continuous": [[5,0,0,0,0]], "N": [0] },
                                "Tile": { "Continuous": [[0,0,5,5]], "N": [15] }
                            },
                            6: {
                                "Entity": { "Continuous": [[6,6,6,6,6]], "N": [8] }, 
                                "Item": { "Continuous": [[6,0,0,0,0]], "N": [1] }, 
                                "Market": { "Continuous": [[6,0,0,0,0]], "N": [0] }, 
                                "Tile": { "Continuous": [[0,0,6,6]], "N": [15] }
                            }
                        }
                    }
        obs = env._obs_team_to_mb_agent(team_obs)
        for k, v in team_obs.items():
            for k1, v1 in team_obs[k].items():
                for k2, v2 in team_obs[k][k1].items():
                    self.assertTrue(team_obs[k][k1][k2] in obs[k][k2])

if __name__ == '__main__':
    unittest.main()