include("../src/task_generator.jl")
using Interact

# mdp, pol, adv_mdp = generate_task(return_orig = true)
# hist1 = simulate(HistoryRecorder(), mdp, pol)
# states1 = collect(state_hist(hist1))
@manipulate for i in 1:length(states1)
    render(mdp, states1[i])
end

# optimal_adv_pol = solve_for_policy(adv_mdp)
# hist2 = simulate(HistoryRecorder(), adv_mdp, optimal_adv_pol)
# states2 = collect(state_hist(hist2))
@manipulate for i in 1:length(states2)
    render(adv_mdp, states2[i])
end

