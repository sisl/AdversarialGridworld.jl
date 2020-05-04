include("gridworld_with_adversary.jl")

using LocalApproximationValueIteration
using GridInterpolations
using LocalFunctionApproximation
using Serialization
using POMDPSimulators

function policy_to_table(mdp, policy)
    lengths = (mdp.size[1], mdp.size[2], mdp.size[1], mdp.size[2])
    policy_table = Array{Symbol, 4}(undef, lengths...)
    for ijk in CartesianIndices(lengths)
        policy_table[ijk.I...] = action(policy, S(ijk.I...))
    end
    policy_table
end

function solve_for_policy(mdp; n_generative_samples = 10, max_iterations = 100, verbose = true)
    # Construct the grid
    Nx, Ny = mdp.size
    grid = RectangleGrid(1:Nx, 1:Ny, 1:Nx, 1:Ny)
    solver = LocalApproximationValueIterationSolver(
                    LocalGIFunctionApproximator(grid),
                    is_mdp_generative = true,
                    n_generative_samples = n_generative_samples,
                    verbose = verbose,
                    max_iterations = max_iterations,
                    belres = 1e-6)
    solve(solver, mdp)
end

function generate_task(;map_rng = MersenneTwister(0),
                        task_rng = MersenneTwister(0),
                        N_rewards = 4,
                        N_walls = 10,
                        Nx = 10,
                        Ny = 10,
                        max_reward = 100,
                        max_penalty = 100,
                        tprob_train = 1,
                        tprob_test = 0.7,
                        n_generative_samples_train = 10,
                        max_iterations_train = 100,
                        verbose = true,
                        return_orig = false)
    rewards = Dict{GWPos, Float64}()
    perturbed_rewards = Dict{GWPos, Float64}()
    for i = 1:N_rewards
        pos = GWPos(rand(map_rng, 1:Nx), rand(map_rng, 1:Ny))
        perturbed_val = rand(task_rng, 1:max_reward)
        rewards[pos] = 1
        perturbed_rewards[pos] = perturbed_val
    end
    perturbed_penalty = rand(task_rng, 1:max_penalty)

    walls = GWPos[]
    while length(walls) < N_walls
        pos = GWPos(rand(map_rng, 1:Nx), rand(map_rng, 1:Ny))
        if !(pos in keys(rewards))
            push!(walls, pos)
        end
    end

    # Create the mdp that will be solved for the policy (perturbed rewards)
    mdp = GridworldAdversary(rewards = perturbed_rewards,
                             walls = walls,
                             tprob = tprob_train,
                             failure_penalty = perturbed_penalty)
    # Solve for the ego policy and store it as an easily accessible table
    policy = solve_for_policy(mdp,
                    n_generative_samples = n_generative_samples_train,
                    max_iterations = max_iterations_train,
                    verbose = verbose)
    policy_table = policy_to_table(mdp, policy)

    # Construct the adversarial mdp with the solved ego policy with a reward that
    # is consistent across tasks
    adv_mdp = GridworldAdversary(rewards = rewards,
                                 walls = walls,
                                 agent_gets_action = :adversary,
                                 tprob = 1,
                                 ego_policy = (s, rng) -> rand(rng) < tprob_test ? policy_table[s...] : rand(rng, actions(mdp)),
                                 failure_penalty = max_penalty)
     if return_orig
         return mdp, policy, adv_mdp
     end
     adv_mdp
end

function generate_task_set(seed, N; folder = nothing, save = false, lzeros = 2)
    try mkdir(folder) catch end
    rng = MersenneTwister(seed)
    tasks = []
    for i=1:N
        println("Generating task ", i, " in ", folder)
        t = generate_task(task_rng = rng, verbose = false)
        push!(tasks, t)
        if save
            name = string(folder, "/task_", lpad(i, lzeros, "0"), ".jls")
            println("writing ", name)
            serialize(name, t)
        end
    end
    tasks
end

function solve_for_policies(tasks; folder = nothing, save = false, lzeros = 2)
    try mkdir(folder) catch end
    policies = []
    for i in 1:length(tasks)
        t = tasks[i]
        pol = solve_for_policy(t, verbose = false)
        push!(policies, pol)
        if save
            name = string(folder, "/policy_", lpad(i, lzeros, "0"), ".jls")
            println("writing ", name)
            serialize(name, pol)
        end
    end
    policies
end

function load_items(folder)
    files = readdir(folder)
    items = []
    for f in files
        i = deserialize(string(folder,"/",f))
        push!(items, i)
    end
    return items
end


function evaluate_policies(tasks, policies, trials, rng::AbstractRNG = Random.GLOBAL_RNG)
    r = 0
    for ti in 1:length(tasks)
        t = tasks[ti]
        policy = policies[ti]
        for i=1:trials
            r += simulate(RolloutSimulator(rng), t, policy)
        end
    end
    r / (length(tasks)*trials)
end

