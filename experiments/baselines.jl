include("../src/gridworld_with_adversary.jl")
include("../src/task_generator.jl")
using POMDPPolicies
using Plots

## Generate training and test sets
N_train = 30
N_test = 5
train_seed = 0
test_seed = 1

# Generate training tasks
train_tasks = generate_task_set(train_seed, N_train, folder = "train_tasks", save = true)
# If the tasks have already been generated then load them
train_tasks = load_task_set("train_tasks")

# Generate a test set of tasks
test_tasks = generate_task_set(test_seed, N_test, folder = "test_tasks", save = true)
# If the test tasks have already been generated then load them
# test_tasks = load_task_set("test_tasks")

## Evaluate different baselines
N_trials = 100

# Evaluate a random policy
random_policies = [RandomPolicy(t) for t in test_tasks]
random_pol_r = evaluate_policies(test_tasks, random_policies, N_trials)
println("random policy reward: ", random_pol_r)

# Evaluate the optimal adversary policy
optimal_policies = [solve_for_policy(t, verbose = false) for t in test_tasks]
optimal_pol_r = evaluate_policies(test_tasks, optimal_policies, N_trials)
println("optimal policy reward: ", optimal_pol_r)

# Evaluate a policy optimized on one of the training tasks
train_task = rand(train_tasks)
train_optimal = solve_for_policy(train_task, verbose = false)
train_optimal_r = evaluate_policies(test_tasks, [train_optimal for t in test_tasks], N_trials)
println("train policy reward: ", train_optimal_r)

x = 1:100
plot(x, (x)->random_pol_r, label="Baseline -- Random Policy", linewidth=3, xlabel = "experience", ylabel="cumulative reward", title = "Test Task Performance vs Experience")
plot!(x, (x)->train_optimal_r, label="Baseline -- Optimal for Random Task", linewidth=3)
plot!(x, (x)->optimal_pol_r, label="Baseline -- Optimal for Test Tasks", linewidth=3)
savefig("baselines.pdf")


