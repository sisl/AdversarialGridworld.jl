include("../src/task_generator.jl")
using Interact

mdp, pol, adv_mdp = generate_task(return_orig = true)
hist = simulate(HistoryRecorder(), mdp, pol)
states = collect(state_hist(hist))
@manipulate for i in 1:length(states)
    render(mdp, states[i])
end

optimal_adv_pol = solve_for_policy(adv_mdp)
hist = simulate(HistoryRecorder(), adv_mdp, optimal_adv_pol)
states = collect(state_hist(hist))
@manipulate for i in 1:length(states)
    render(adv_mdp, states[i])
end

