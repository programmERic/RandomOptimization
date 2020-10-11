import mlrose
import numpy as np
SEED=np.random.seed(12)
import time
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def MaximizeOnesExperiment():
    fitness = mlrose.OneMax()

    schedule = mlrose.ExpDecay()

    max_attempts = 100
    max_iters = np.inf
    
    pop_size = 100
    keep_pct = 0.1

    bit_string_length = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    score = np.zeros(shape=(4,len(bit_string_length)))
    runtime = np.zeros(shape=(4,len(bit_string_length)))
    iterations = np.zeros(shape=(4,len(bit_string_length)))

    trials = 1

    experiment_start = time.time()
    
    for run, length in enumerate(bit_string_length):

        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

        for trial in range(trials):

            init_state = np.random.randint(0, high=2, size=(length,))#np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])


            rhc_start = time.time()
            rhc_out = mlrose.random_hill_climb(problem, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED)
            runtime[0,run] += (time.time() - rhc_start)/trials
            score[0,run] += rhc_out[1]/trials
            iterations[0,run] += rhc_out[2]/trials
            

            ga_start = time.time()
            ga_out = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[1,run] += (time.time() - ga_start)/trials
            score[1,run] += ga_out[1]/trials
            iterations[1,run] += ga_out[2]/trials
            

            sa_start = time.time()
            sa_out = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED)
            runtime[2,run] += (time.time() - sa_start)/trials
            score[2,run] += sa_out[1]/trials
            iterations[2,run] += sa_out[2]/trials
            

            mimic_start = time.time()
            mimic_out = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            score[3,run] += mimic_out[1]/trials
            runtime[3,run] += (time.time() - mimic_start)/trials
            iterations[3,run] += mimic_out[2]/trials
            

            print(run, trial)

    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    labels = ["RHC", "GA", "SA", "MIMIC"]

    print(score)
    plt.figure()
    plt.title("Best Fitness Score Obtained (Maximize Ones)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Score")
    for i, label in enumerate(labels):
        plt.plot(bit_string_length, score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("maxones_score.png")

    print(runtime)
    plt.figure()
    plt.title("Average Algorithm Runtime (Maximize Ones)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Time (s)")
    for i, label in enumerate(labels):
        plt.semilogy(bit_string_length, runtime[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("maxones_time.png")

    print(iterations)
    plt.figure()
    plt.title("Average # of Iterations per Algorithm Run (Maximize Ones)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Iterations")
    for i, label in enumerate(labels):
        plt.semilogy(bit_string_length, iterations[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("maxones_iterations.png")

    # 
    plt.figure()
    plt.title("Max Fitness Score Achieved Over Algorithm Iterations (Maximize Ones)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness Score")
    for i, label in enumerate(labels):
        plt.semilogx(iterations[i,:], score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("maxones_score_v_iterations.png")

def MaximizeOnesExperimentTestTempDecay():
    fitness = mlrose.OneMax()

    exp_decay = mlrose.ExpDecay()
    arith_decay = mlrose.ArithDecay()
    geom_decay = mlrose.GeomDecay()

    max_attempts = 1000
    max_iters = 1000
    
    pop_size = 100
    keep_pct = 0.1

    bit_string_length = 100
    curve = True
    trials = 1

    experiment_start = time.time()
    
    problem = mlrose.DiscreteOpt(length=bit_string_length, fitness_fn=fitness, maximize=True, max_val=2)

    init_state = np.random.randint(0, high=2, size=(bit_string_length,))

    sa_start = time.time()
    sa_out = mlrose.simulated_annealing(problem, schedule=exp_decay, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED, curve=curve)
    exp_decay_curve = sa_out[2]

    sa_start = time.time()
    sa_out = mlrose.simulated_annealing(problem, schedule=arith_decay, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED, curve=curve)
    arith_decay_curve = sa_out[2]

    sa_start = time.time()
    sa_out = mlrose.simulated_annealing(problem, schedule=geom_decay, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED, curve=curve)
    geom_decay_curve = sa_out[2]


    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    plt.figure()
    plt.title("Analysis of Types of Temperature Decay SA on Score (Maximize Ones)")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    
    plt.plot(exp_decay_curve, label="Exponential")
    plt.plot(arith_decay_curve, label="Arithmetic")
    plt.plot(geom_decay_curve, label="Geometric")
    
    plt.legend(loc="best")
    plt.savefig("maxones_SA_decay_analysis.png")



def FourPeaksExperimentTestPopulationSize():
    
    fitness = mlrose.FourPeaks(t_pct=0.1)

    schedule = mlrose.ExpDecay()

    max_attempts = 10
    max_iters = np.inf
    
     
    keep_pct = 0.1

    bit_string_length = 50
    population_sizes = [2, 10, 50, 100, 200, 500, 1000]
    score = np.zeros(shape=(2,len(population_sizes)))
    runtime = np.zeros(shape=(2,len(population_sizes)))
    iterations = np.zeros(shape=(2,len(population_sizes)))

    trials = 5

    experiment_start = time.time()
    
    for run, pop_size in enumerate(population_sizes):

        problem = mlrose.DiscreteOpt(length=bit_string_length, fitness_fn=fitness, maximize=True, max_val=2)

        for trial in range(trials):
            
            ga_start = time.time()
            ga_out = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[0,run] += (time.time() - ga_start)/trials
            score[0,run] += ga_out[1]/trials
            iterations[0,run] += ga_out[2]/trials


            mimic_start = time.time()
            mimic_out = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            score[1,run] += mimic_out[1]/trials
            runtime[1,run] += (time.time() - mimic_start)/trials
            iterations[1,run] += mimic_out[2]/trials

            print(run, trial)

    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    labels = ["GA", "MIMIC"]

    print(score)
    plt.figure()
    plt.title("Analysis of Population Size Parameter on Score (Four Peaks)")
    plt.xlabel("Population Size")
    plt.ylabel("Score")
    for i, label in enumerate(labels):
        plt.plot(population_sizes, score[i,:], label=label+" Score")
        plt.plot(population_sizes, runtime[i,:], label=label+" Runtime")
    plt.legend(loc="best")
    plt.savefig("fourpeaks_population_size_analysis.png")

    

    
def FourPeaksExperiment():

    fitness = mlrose.FourPeaks(t_pct=0.1)

    schedule = mlrose.ExpDecay()

    max_attempts = 10
    max_iters = np.inf
    
     
    keep_pct = 0.1

    bit_string_length = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    score = np.zeros(shape=(4,len(bit_string_length)))
    runtime = np.zeros(shape=(4,len(bit_string_length)))
    iterations = np.zeros(shape=(4,len(bit_string_length)))

    trials = 10

    experiment_start = time.time()
    
    for run, length in enumerate(bit_string_length):

        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

        for trial in range(trials):

            pop_size = 2*length
            init_state = np.random.randint(0, high=2, size=(length,))#np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])

            rhc_start = time.time()
            rhc_out = mlrose.random_hill_climb(problem, max_attempts=max_attempts, max_iters=max_iters, init_state=init_state, random_state=SEED)
            runtime[0,run] += (time.time() - rhc_start)/trials
            score[0,run] += rhc_out[1]/trials
            iterations[0,run] += rhc_out[2]/trials
            
            ga_start = time.time()
            ga_out = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[1,run] += (time.time() - ga_start)/trials
            score[1,run] += ga_out[1]/trials
            iterations[1,run] += ga_out[2]/trials
            
            
            sa_start = time.time()
            sa_out = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[2,run] += (time.time() - sa_start)/trials
            score[2,run] += sa_out[1]/trials
            iterations[2,run] += sa_out[2]/trials

            mimic_start = time.time()
            mimic_out = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            score[3,run] += mimic_out[1]/trials
            runtime[3,run] += (time.time() - mimic_start)/trials
            iterations[3,run] += mimic_out[2]/trials

            print(run, trial)

    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    labels = ["RHC", "GA", "SA", "MIMIC"]

    print(score)
    plt.figure()
    plt.title("Best Fitness Score Obtained (Four Peaks)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Score")
    for i, label in enumerate(labels):
        plt.plot(bit_string_length, score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("fourpeaks_score.png")

    print(runtime)
    plt.figure()
    plt.title("Average Algorithm Runtime (Four Peaks)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Time (s)")
    for i, label in enumerate(labels):
        plt.semilogy(bit_string_length, runtime[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("fourpeaks_time.png")

    print(iterations)
    plt.figure()
    plt.title("Average # of Iterations per Algorithm Run (Four Peaks)")
    plt.xlabel("Bit String Length")
    plt.ylabel("Iterations")
    for i, label in enumerate(labels):
        plt.semilogy(bit_string_length, iterations[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("fourpeaks_iterations.png")

    plt.figure()
    plt.title("Max Fitness Score Achieved Over Algorithm Iterations (Four Peaks)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness Score")
    for i, label in enumerate(labels):
        plt.semilogx(iterations[i,:], score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("fourpeaks_score_v_iterations.png")


def heart_failure_data(filename='heart_failure_clinical_records_dataset.csv'):
    '''
    Heart Disease
    '''
    heart_failure = np.genfromtxt(filename, delimiter=',')

    heart_failure = heart_failure[1:, :]

    heart_X = heart_failure[:, :-1]
    heart_y = np.reshape(heart_failure[:, -1], (-1, 1))
    
    return heart_X, heart_y    

def NeuralNetworkExperiment():

    X, y = heart_failure_data()
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.4, random_state=SEED)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

    
    max_attempts = 100
    max_iters = 10000#np.inf

    learning_rate = 0.0001
    bias = True
    is_classifier = True
    early_stopping = True
    clip_max = 5

    hidden_nodes = [2,3]

    nn_model_rhc = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', algorithm='random_hill_climb', max_iters=max_iters, bias=bias, is_classifier=is_classifier,
                                        learning_rate=learning_rate, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=SEED, curve=True)

    rhc_fit_time = time.time()
    nn_model_rhc.fit(X_train_scaled, y_train_hot)
    print("RHC Fit time", time.time()-rhc_fit_time)
    y_train_pred = nn_model_rhc.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model_rhc.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print("RHC", "Train", y_train_accuracy, "Test", y_test_accuracy)
    plot_learning_curve(nn_model_rhc, X_test_scaled, y_test_pred, name="NN Learning Curve with Weights set by RHC", cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))


    nn_model_sa = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', algorithm='simulated_annealing', max_iters=max_iters, bias=bias, is_classifier=is_classifier,
                                        learning_rate=learning_rate, early_stopping=early_stopping, clip_max=5, max_attempts=max_attempts, random_state=SEED, curve=True)

    sa_fit_time = time.time()
    nn_model_sa.fit(X_train_scaled, y_train_hot)
    print("SA Fit time", time.time()-sa_fit_time)
    y_train_pred = nn_model_sa.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model_sa.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print("SA", "Train", y_train_accuracy, "Test", y_test_accuracy)
    plot_learning_curve(nn_model_sa, X_test_scaled, y_test_pred, name="NN Learning Curve with Weights set by SA", cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))

    
    nn_model_ga = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', algorithm='genetic_alg', max_iters=max_iters, bias=bias, is_classifier=is_classifier,
                                        learning_rate=learning_rate, early_stopping=early_stopping, clip_max=5, max_attempts=max_attempts, random_state=SEED, curve=True)    

    ga_fit_time = time.time()
    nn_model_ga.fit(X_train_scaled, y_train_hot)
    print("GA Fit time", time.time()-ga_fit_time)
    y_train_pred = nn_model_ga.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model_ga.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print("GA", "Train", y_train_accuracy, "Test", y_test_accuracy)
    plot_learning_curve(nn_model_ga, X_test_scaled, y_test_pred, name="NN Learning Curve with Weights set by GA", cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))


    nn_model_baseline = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', algorithm='gradient_descent', max_iters=max_iters, bias=bias, is_classifier=is_classifier,
                                        learning_rate=learning_rate, early_stopping=early_stopping, clip_max=5, max_attempts=max_attempts, random_state=SEED, curve=True)    

    baseline_fit_time = time.time()
    nn_model_baseline.fit(X_train_scaled, y_train_hot)
    print("Baseline Fit time", time.time()-baseline_fit_time)
    y_train_pred = nn_model_baseline.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    y_test_pred = nn_model_baseline.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    print("Baseline", "Train", y_train_accuracy, "Test", y_test_accuracy)
    plot_learning_curve(nn_model_baseline, X_test_scaled, y_test_pred, name="NN Learning Curve with Weights set by Gradient Descent", cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10))


    plot_fitness_curves(nn_model_rhc, nn_model_sa, nn_model_ga, nn_model_baseline)

def plot_fitness_curves(nn_model_rhc, nn_model_sa, nn_model_ga, nn_model_baseline):

        plt.figure()
        _, axes = plt.subplots(2, 2, figsize=(20, 20))

        _.suptitle('NN Fitness Score versus Algorithm Iterations', fontsize='xx-large')

        axes[0,0].plot(nn_model_rhc.fitness_curve, color='blue', lw=1.25)
        axes[0,0].set_title('RHC')
        axes[0,1].plot(nn_model_sa.fitness_curve, color='green', lw=1.25)
        axes[0,1].set_title('SA')
        axes[1,0].plot(nn_model_ga.fitness_curve, color='orange', lw=1.25)
        axes[1,0].set_title('GA')
        axes[1,1].plot(nn_model_baseline.fitness_curve, color='purple', lw=1.25)
        axes[1,1].set_title('Gradient Descent')

        plt.savefig("model_fitness_curves.png")


def plot_learning_curve(nn_model, X, y, name, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
        
        estimator = nn_model

        plt.figure()
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(name)
        axes[0].set_ylim((0., 1.1))
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(nn_model, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        plt.savefig(name+"_learning_curve.png")
    
def KnapSackExperiment():

    weights = [2,3,5,7,1,4,1]
    values = [10,5,15,7,6,18,3]
    fitness = mlrose.Knapsack(weights=weights, values=values, max_weight_pct=0.5)
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.array([0,0,0,1,1,0,1])


    schedule = mlrose.ExpDecay()

    max_attempts = 10
    max_iters = np.inf
    
     
    keep_pct = 0.1
    pop_size = 100

    trials = 10

    score = np.zeros(shape=(4,trials))
    runtime = np.zeros(shape=(4,trials))
    iterations = np.zeros(shape=(4,trials))

    experiment_start = time.time()

    knapsack_sizes = [4, 10, 25, 50, 100]

    score = np.zeros(shape=(4,len(knapsack_sizes)))
    runtime = np.zeros(shape=(4,len(knapsack_sizes)))
    iterations = np.zeros(shape=(4,len(knapsack_sizes)))
    
    for run, size in enumerate(knapsack_sizes):

        weights = np.random.randint(1, high=size//2, size=(size,))#[2,3,5,7,1,4,1]
        values = np.random.randint(1, high=size//2, size=(size,))#[10,5,15,7,6,18,3]
        fitness = mlrose.Knapsack(weights=weights, values=values, max_weight_pct=0.5)
        problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)

        for trial in range(trials):

            rhc_start = time.time()
            rhc_out = mlrose.random_hill_climb(problem, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[0,run] += (time.time() - rhc_start)/trials
            score[0,run] += rhc_out[1]/trials
            iterations[0,run] += rhc_out[2]/trials
            
            ga_start = time.time()
            ga_out = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[1,run] += (time.time() - ga_start)/trials
            score[1,run] += ga_out[1]/trials
            iterations[1,run] += ga_out[2]/trials
            
            
            sa_start = time.time()
            sa_out = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            runtime[2,run] += (time.time() - sa_start)/trials
            score[2,run] += sa_out[1]/trials
            iterations[2,run] += sa_out[2]/trials

            mimic_start = time.time()
            mimic_out = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED)
            score[3,run] += mimic_out[1]/trials
            runtime[3,run] += (time.time() - mimic_start)/trials
            iterations[3,run] += mimic_out[2]/trials

            print(run, trial)

        
    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    labels = ["RHC", "GA", "SA", "MIMIC"]

    print(score)
    plt.figure()
    plt.title("Best Fitness Score Obtained (Knapsack)")
    plt.xlabel("Knapsack size")
    plt.ylabel("Score")
    for i, label in enumerate(labels):
        plt.plot(knapsack_sizes, score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("knapsack_score.png")

    print(runtime)
    plt.figure()
    plt.title("Average Algorithm Runtime (Knapsack)")
    plt.xlabel("Knapsack size")
    plt.ylabel("Time (s)")
    for i, label in enumerate(labels):
        plt.semilogy(knapsack_sizes, runtime[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("knapsack_time.png")

    print(iterations)
    plt.figure()
    plt.title("Average # of Iterations per Algorithm Run (Knapsack)")
    plt.xlabel("Knapsack size")
    plt.ylabel("Iterations")
    for i, label in enumerate(labels):
        plt.semilogy(knapsack_sizes, iterations[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("knapsack_iterations.png")

    # 
    plt.figure()
    plt.title("Max Fitness Score Achieved Over Algorithm Iterations (Knapsack)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness Score")
    for i, label in enumerate(labels):
        plt.semilogx(iterations[i,:], score[i,:], label=label)
    plt.legend(loc="best")
    plt.savefig("knapsack_score_v_iterations.png")


def KnapSackExperimentKeepPctEval():

    weights = [2,3,5,7,1,4,1]
    values = [10,5,15,7,6,18,3]
    fitness = mlrose.Knapsack(weights=weights, values=values, max_weight_pct=0.5)
    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)
    init_state = np.array([0,0,0,1,1,0,1])


    max_attempts = 10
    max_iters = np.inf
     
    pop_size = 100

    trials = 1

    score = np.zeros(shape=(4,trials))
    runtime = np.zeros(shape=(4,trials))
    iterations = np.zeros(shape=(4,trials))

    experiment_start = time.time()

    knapsack_size = 50

    keep_percentages = [0.125, 0.25, 0.5, 0.75, 1.0]

    score = []

    weights = np.random.randint(1, high=knapsack_size//2, size=(knapsack_size,))
    values = np.random.randint(1, high=knapsack_size//2, size=(knapsack_size,))
    
    for run, keep_pct in enumerate(keep_percentages):

        fitness = mlrose.Knapsack(weights=weights, values=values, max_weight_pct=0.5)
        problem = mlrose.DiscreteOpt(length=knapsack_size, fitness_fn=fitness, maximize=True, max_val=2)

        for trial in range(trials):

            mimic_start = time.time()
            mimic_out = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=max_iters, random_state=SEED, fast_mimic=True, curve = True)
            score.append(mimic_out[2])

            print(run, trial)

        
    print("Total Experiment Elapsed Time:", time.time()-experiment_start)

    labels = ["MIMIC"]

    print(score)
    plt.figure()
    plt.title("Analysis of 'Keep %' parameter of MIMIC (Knapsack)")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    for i, label in enumerate(keep_percentages):
        plt.plot(score[i], label=label)
    plt.legend(loc="best")
    plt.savefig("knapsack_keep_pct_analysis.png")




if __name__ == "__main__":

    NeuralNetworkExperiment()







    
