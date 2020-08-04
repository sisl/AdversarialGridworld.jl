# Computes the average undiscounted perfromance of a policy
function average_performance(t, p, Neps)
    vals = zeros(Neps)
    for i = 1:Neps
        hist = simulate(HistoryRecorder(), t, p)
        vals[i] = undiscounted_reward(hist)
    end
    mean(vals), std(vals) / sqrt(Neps)
end

# Converts a policy to a table that maps states to actions
function policy_to_table(mdp, policy)
    lengths = (mdp.size[1], mdp.size[2], mdp.size[1], mdp.size[2])
    policy_table = Array{Symbol, 4}(undef, lengths...)
    for ijk in CartesianIndices(lengths)
        policy_table[ijk.I...] = action(policy, S(ijk.I...))
    end
    policy_table
end

# Gets a vector of action values for a given state using POMDPs.value(pol, s, a)
action_values(pol, s) = [value(pol, s, a) for a in actions(pol.mdp)]

# Converts a policy to a table that maps state,action pairs to value
# relies on POMDPs.value(pol, s, a)
function value_to_table(policy)
    mdp = policy.mdp
    lengths = (mdp.size[1], mdp.size[2], mdp.size[1], mdp.size[2])
    policy_table = Array{Float64, 5}(undef, lengths..., length(actions(mdp)))
    for ijk in CartesianIndices(lengths)
        policy_table[ijk.I..., :] .= action_values(policy, S(ijk.I...))
    end
    policy_table
end

# Solves the mdp for the optimal policy using LocalApproximation value iteration
function solve_for_policy(mdp; n_generative_samples = 10, max_iterations = 100, verbose = true)
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

# Generates a gridworld_with_adversary task
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
        rewards[pos] = 1
        perturbed_rewards[pos] = rand(task_rng, 1:max_reward)
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
    mdp = AdversarialGridworldMDP(rewards = perturbed_rewards,
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
    adv_mdp = AdversarialGridworldMDP(rewards = rewards,
                                 walls = walls,
                                 agent_gets_action = :adversary,
                                 tprob = 1,
                                 ego_policy = (s, rng) -> rand(rng) < tprob_test ? policy_table[s...] : rand(rng, actions(mdp)),
                                 failure_penalty = 1)
     if return_orig
         return mdp, policy_table, adv_mdp
     end
     adv_mdp
end

# Constructs (and optionally saves) a set of tasks using a specified random seed
function generate_task_set(seed, N; tprob_test = 0.7, folder = nothing, save = false, lzeros = 2, keep_map = true, verbose = false)
    try mkdir(folder) catch end
    task_rng = MersenneTwister(seed)
    map_rng = MersenneTwister(seed)
    tasks = []
    for i=1:N
        println("Generating task ", i, " in ", folder)
        mdp, policy_table, advmdp = generate_task(task_rng = task_rng, map_rng = keep_map ? MersenneTwister(0) : map_rng, verbose = verbose, tprob_test = tprob_test, return_orig = true)
        push!(tasks, advmdp)
        if save
            subfolder = string(folder, "/task_", lpad(i, lzeros, "0"),"/")
            try mkdir(subfolder) catch end
            serialize(string(subfolder, "mdp"), mdp)
            serialize(string(subfolder, "policy_table"), policy_table)
            serialize(string(subfolder, "advmdp"), advmdp)
            draw(PDF(string(subfolder, "mdp.pdf")), POMDPModelTools.render(mdp, initialstate(mdp)))
            draw(PDF(string(subfolder, "advmdp.pdf")), POMDPModelTools.render(advmdp, initialstate(advmdp)))
        end
    end
    tasks
end

# Solves for (and optionally saves) the optimal policy of a set of tasks
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

# converts (and optionally saves) a set of policies to tables
function convert_policies_to_tables(policies; folder = nothing, save = false, lzeros = 2)
    try mkdir(folder) catch end
    tables = []
    for i in 1:length(policies)
        tab = value_to_table(policies[i])
        push!(tables, tab)
        if save
            name = string(folder, "/table_", lpad(i, lzeros, "0"), ".jls")
            println("writing ", name)
            serialize(name, tab)
        end
    end
    tables
end

# Deserializes a folder of items
function load_items(folder)
    files = readdir(folder)
    items = []
    for f in files
        i = deserialize(string(folder,"/",f))
        push!(items, i)
    end
    return items
end
