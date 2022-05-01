import neat

class DerkNeatNNPlayer:
    def __init__(self, n_agents: int, action_space: int, neat_config_path: str):
        self.n_agents = n_agents
        self.action_space = action_space
        self.genome = neat.DefaultGenome
        self.config = neat.Config(
            self.genome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path
        )
        self.network = neat.nn.FeedForwardNetwork.create(neat.DefaultGenome, self.config)

    def evolve(self, rewards):
        self.best = self.population.run()

    def load_params(self, path: str):
        # np.load(path)
        pass

    def dump_params(self, path: str):
        # np.save("weights_path", top_network.weights)
        pass

    def signal_env_reset(self, obs):
        pass

    def take_action(self, env_step_ret):
        pass


# # set the number of episodes
# episodes = 10
# for e in range(episodes):
#     # reset the environment at each episode
#     observation_n = env.reset()
#     while True:
#         # get the actions from the networks
#         action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
#         # update
#         observation_n, reward_n, done_n, info = env.step(action_n)

#         if all(done_n):
#             print("Episode finished")
#             break

#     # this is the part that needs to be changed with neat
#     if env.mode == "train":
#         reward_n = env.total_reward
#         print(reward_n)
#         top_network_i = np.argmax(reward_n)
#         top_network = networks[top_network_i].clone()
#         for network in networks:
#             network.copy_and_mutate(top_network)
#         print("top reward", reward_n[top_network_i])

#         # save network
#         np.save("weights_path", top_network.weights)
#         np.save("biases_path", top_network.biases)
