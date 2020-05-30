using AdversarialGridworld
using Test
using Random

# Generate a sample task
mdp, policy, adv_mdp = generate_task(N_rewards=1, N_walls=1, Nx=4, Ny=4,
                                     return_orig=true, n_generative_samples_train=1,
                                     max_iterations_train=10)


@test mdp.agent_gets_action == :ego
@test adv_mdp.agent_gets_action == :adversary
