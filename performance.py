from utils.rl_tools import eval_trained_model, test_trained_agent

# Get two models
# Evaluate the model
eval_trained_model("logs/best_models/1.zip", "procgen:procgen-bigfish-v0", 10)
eval_trained_model("logs/best_models/best_model.zip", "procgen:procgen-bigfish-v0", 10)

# Test the agent
mean_max_reward1 = test_trained_agent("logs/best_models/1.zip","procgen:procgen-bigfish-v0", 10)
mean_max_reward2 = test_trained_agent("logs/best_models/best_model.zip","procgen:procgen-bigfish-v0", 10)

print('Mean reward for model 1: ', mean_max_reward1['Mean_reward'])
print('Mean reward for model 2: ', mean_max_reward2['Mean_reward'])