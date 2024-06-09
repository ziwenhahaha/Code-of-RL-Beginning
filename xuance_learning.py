import xuance
# import gymnasium as gym


runner = xuance.get_runner(method = 'dqn',
                        env = 'classic_control',
                        env_id = 'CartPole-v1',
                        config_path= "./test.yaml",
                        is_test = False,)


runner.run()
