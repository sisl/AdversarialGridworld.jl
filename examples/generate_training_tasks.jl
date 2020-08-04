using AdversarialGridworld

## Generate training and test sets
N_train = 30
seed = 0
tprob_test = 0.7
folder = "data"

pwd()
cd(folder)

# Generate training tasks
train_tasks = generate_task_set(seed, N_train, tprob_test = tprob_test, folder = "diffmaps", save = true, keep_map = false, verbose = true)

