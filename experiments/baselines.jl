include("../src/task_generator.jl")
include("../src/a2t.jl")

# using Plots
using DeepQLearning
using Flux: glorot_normal
using POMDPModels
using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations
using POMDPs
using POMDPPolicies

## Generate training and test sets
N_train = 30
N_test = 5
train_seed = 0
test_seed = 1
tprob_test = 0.7
folder = "data/random_transitions"

pwd()
cd(folder)

# Generate training tasks
# train_tasks = generate_task_set(train_seed, N_train, tprob_test = tprob_test, folder = "train_tasks", save = true)
# If the tasks have already been generated then load them
train_tasks = load_items("train_tasks")

# Solve for training policies
# train_policies = solve_for_policies(train_tasks, folder = "train_policies", save = true)
# If tasks have already been solved, load the policies
train_policies = load_items("train_policies")
# train_tables = convert_policies_to_tables(train_policies, folder = "train_policy_tables", save = true)
train_tables = load_items("train_policy_tables/")

# Generate a test set of tasks
# test_tasks = generate_task_set(test_seed, N_test, tprob_test = tprob_test, folder = "test_tasks", save = true)
# If the test tasks have already been generated then load them
test_tasks = load_items("test_tasks")

# Solve for test optimal policies
# test_policies = solve_for_policies(test_tasks, folder = "test_policies", save = true)
# If test tasks have already been soved, load the policies
test_policies = load_items("test_policies")
# test_tables = convert_policies_to_tables(test_policies, folder = "test_policy_tables", save = true)
test_tables = load_items("test_policy_tables/")

## Setup params
i = 2
train_task = deepcopy(train_tasks[i]) # The train task learn
test_task = deepcopy(test_tasks[i]) # The test task to learn
Na = length(actions(train_task)) # Number of actions
Np = length(train_tasks) # Number of train policies to be used
Ni = length(initialstate(train_task))
scaling = (x) -> (x .- 5) ./ 5. # Scales the input to be between -1, 1
solutions = [(x) -> x[1] < 0 ? zeros(Na) : tab[Int.(x)..., :] for tab in train_tables] # Computes the solutions from the train task
Neps = 100

## The task we are going to solve
t = train_task
opt_policy = train_policies[i]
prefix = "train"

## Bounds on performance
lb, lb_std = average_performance(t, RandomPolicy(t), Neps)
ub, ub_std = average_performance(t, opt_policy, Neps)
serialize(string(prefix, "_bounds.jls"), (lb, lb_std, ub, ub_std))

## Baseline - DQN from scratch
Nstep = 2000000
model = Chain(scaling, Dense(Ni, 32, relu), Dense(32, 32, relu), Dense(32, Na))
solver = DeepQLearningSolver(qnetwork = model,
                             num_ep_eval = Neps,
                             target_update_freq = 1000,
                             dueling = false,
                             exploration_policy = EpsGreedyPolicy(t, LinearDecaySchedule(start=1.0, stop=0.01, steps=Nstep/2)),
                             max_steps = Nstep,
                             max_episode_length = 1000,
                             logdir = string("log/", prefix, "_baseline"),
                             learning_rate=.0001,)
policy = solve(solver, t)
baseline, baseline_std = average_performance(t, policy, Neps)

## A2t approach With state-dependent weights
Nstep = 200000
base = Chain(scaling, Dense(4, 32, relu), Dense(32, Na))
attn = Chain(scaling, Dense(4, 32, relu), Dense(32, Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)
solver = DeepQLearningSolver(qnetwork = a2t_model,
                             num_ep_eval = Neps,
                             dueling = false,
                             eval_freq = 32,
                             log_freq = 32,
                             exploration_policy = EpsGreedyPolicy(t, LinearDecaySchedule(start=1.0, stop=0.01, steps=Nstep/2)),
                             max_steps = Nstep,
                             logdir = string("log/", prefix, "_A2T"),
                             max_episode_length = 1000,
                             learning_rate=.0001,)
policy = solve(solver, t)


## A2T approach with constant weights
Nstep = 200000
base = Chain(scaling, Dense(4, 32, relu), Dense(32, Na))
attn = Chain(ConstantLayer(Np+1), softmax)
a2t_model = A2TNetwork(base, attn, solutions)
solver = DeepQLearningSolver(qnetwork = a2t_model,
                             num_ep_eval = Neps,
                             dueling = false,
                             eval_freq = 32,
                             log_freq = 32,
                             exploration_policy = EpsGreedyPolicy(t, LinearDecaySchedule(start=1.0, stop=0.01, steps=Nstep/2)),
                             max_steps = Nstep,
                             logdir = string("log/", prefix, "_A2T_ConstantW_highlr"),
                             max_episode_length = 1000,
                             learning_rate=.01,)
policy = solve(solver, t)

a2t_model.attn



## Measure the performance of the algorithms on a new task
# plot(title = "Seen Task Performance", legend=:bottomright)
# plot!(tr_exp_pg, [lb_train for i=1:length(tr_exp_pg)], label = "Random Policy")
# plot!(tr_exp_pg, [ub_train for i=1:length(tr_exp_pg)], label = "Optimal Policy")
# plot!(tr_exp_pg, tr_perf_pg, label = "Policy Gradient")
# plot!(tr_exp_Q, tr_perf_Q, label = "Q Learning (1 step return)")
# plot!(tr_exp_MC, tr_perf_MC, label = "Monte Carlo")
#
# savefig("train_task_perf.pdf")
#
# plot(title = "Coefficient of correct train task", legend = :bottomright)
# plot!(tr_exp_pg, [p.μ[1] for p in tr_pol_pg], label="Policy Gradient")
# plot!(tr_exp_Q, [p.w.θ[1] for p in tr_pol_Q], label="Q Learning")
# plot!(tr_exp_Q, [p.w.θ[1] for p in tr_pol_MC], label="MC Learning")
#
# savefig("train_parameters.pdf")
# plot(tr_exp_pg, [p.σ2[1] for p in tr_pol_pg], label="Policy Gradient")
#
#
# plot(title = "Unseen Task Performance", legend=:bottomright)
# plot!(te_exp_pg, [lb_test for i=1:length(te_exp_pg)], label = "Random Policy")
# plot!(te_exp_pg, [ub_test for i=1:length(te_exp_pg)], label = "Optimal Policy")
# plot!(te_exp_pg, te_perf_pg, label = "Policy Gradient")
# plot!(te_exp_Q, te_perf_Q, label = "Q Learning (1 step return)")
# plot!(te_exp_MC, te_perf_MC, label = "Monte Carlo")
#
# savefig("test_task_perf.pdf")

