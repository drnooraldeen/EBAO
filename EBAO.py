import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load KDDCup99 dataset
kddcup99_data = pd.read_csv('KDDCup99.csv')
# Convert non-numeric columns to numeric using one-hot encoding
kddcup99_numeric = pd.get_dummies(kddcup99_data, drop_first=True)

# Split features and labels
X = kddcup99_numeric.iloc[:, :-1].values  # Features
y = kddcup99_numeric.iloc[:, -1].values  # Label

def levy_flight(lamda):
    sigma = (math.gamma(1 + lamda) * math.sin(math.pi * lamda / 2) /
             (math.gamma((1 + lamda) / 2) * lamda * 2**((lamda - 1) / 2)))**(1 / lamda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v)**(1 / lamda)
    return step

def apply_binary_transfer_function(X):
    return 1 / (1 + np.exp(-X))

def convert_to_binary(X):
    return (X > 0.5).astype(int)

def fitness_function(solutions, X, y):
    num_solutions = solutions.shape[0]
    fitness_values = np.zeros(num_solutions)
    
    for i in range(num_solutions):
        selected_features = np.where(solutions[i] == 1)[0]
        if len(selected_features) == 0:
            fitness_values[i] = float('inf')
            continue
        
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        num_selected_features = len(selected_features)
        fitness_values[i] = -1 * (accuracy + f1) / num_selected_features
        print(fitness_values)
    return fitness_values


def binary_bhc(X, beta=0.1, bw=0.01, max_iter=10): 
    # N(X) generates a new feasible solution by adding a uniform noise (bw) to X
    def N(X):
        return X + np.random.uniform(-1, 1, X.shape) * bw

    itr = 1
    while itr <= max_iter:
        X_prime = N(X)  # Create new solution by perturbing X
        for i in range(len(X)):  # Iterate through each element in X
            if np.random.rand() <= beta:  # Mutate with probability beta
                X_prime[i] = X[i] + np.random.uniform(-1, 1) * bw  # Apply mutation to X_i
        X_prime = convert_to_binary(apply_binary_transfer_function(X_prime))  # Convert X_prime to binary

        # Evaluate fitness of new and current solutions
        if fitness_function(np.array([X_prime]), X, y)[0] < fitness_function(np.array([X]), X, y)[0]:
            X = X_prime  # Update X if the new solution is better
        itr += 1
    return X


def hho_mutation(X, X_best, X_rand):
    j = np.random.randint(0, len(X))
    return X + np.random.rand() * (X_best - X) + np.random.rand() * (X_rand - X[j])

def ebao(n, T, beta=0.1, bw=0.01):
    # Initialize the number of features based on input data shape
    num_features = X.shape[1]
    
    # Step 2: Generate initial population
    population = np.random.rand(n, num_features)  # Randomly generate initial population for the first half
    for i in range(n // 2, n):  # For the second half, apply Lévy flight modification
        population[i] = population[i - n // 2] + levy_flight(1.5) * np.random.randn()
    
    # Convert the continuous population to binary form using a binary transfer function (BTF)
    population = convert_to_binary(apply_binary_transfer_function(population))
    
    # Calculate fitness for the initial population
    fitness_values = fitness_function(population, X, y)
    X_best = population[np.argmin(fitness_values)]  # Set the initial best solution
    
    # Iteration counter
    t = 1
    while t <= T:  # Step 4: Main optimization loop until maximum iteration
        # Recalculate fitness and update the best solution for this iteration
        fitness_values = fitness_function(population, X, y)
        X_best = population[np.argmin(fitness_values)]
        
        for i in range(n):  # Step 7: Iterate over each individual in the population
            X_mean = np.mean(population, axis=0)  # Compute the mean of the population
            rand1 = np.random.rand()  # Random values for probabilistic exploration/exploitation
            rand2 = np.random.rand()
            
            if t <= 2 * T / 3:  # Expanded exploration phase (Step 12)
                if rand1 <= 0.5:
                    # Update solution based on proximity to best and mean solutions (Step 13)
                    X1_new = population[i] + np.random.rand() * (X_best - population[i]) + np.random.rand() * (X_mean - population[i])
                    X1_new = convert_to_binary(apply_binary_transfer_function(X1_new))  # Apply BTF
                    
                    # Accept new solution if fitness improves
                    if fitness_function(np.array([X1_new]), X, y)[0] < fitness_function(np.array([population[i]]), X, y)[0]:
                        population[i] = X1_new
                        # Update global best if necessary
                        if fitness_function(np.array([X1_new]), X, y)[0] < fitness_function(np.array([X_best]), X, y)[0]:
                            X_best = X1_new
                else:
                    # Narrowed exploration using Lévy flight (Step 21)
                    X2_new = population[i] + np.random.rand() * (X_best - population[i]) + levy_flight(1.5) * np.random.randn()
                    X2_new = convert_to_binary(apply_binary_transfer_function(X2_new))
                    if fitness_function(np.array([X2_new]), X, y)[0] < fitness_function(np.array([population[i]]), X, y)[0]:
                        population[i] = X2_new
                        if fitness_function(np.array([X2_new]), X, y)[0] < fitness_function(np.array([X_best]), X, y)[0]:
                            X_best = X2_new
            else:  # Exploitation phase (Step 30)
                if rand1 <= 0.5:
                    # Expanded exploitation (Step 31)
                    X3_new = population[i] + np.random.rand() * (X_best - population[i]) + np.random.rand() * (X_mean - population[i])
                    X3_new = convert_to_binary(apply_binary_transfer_function(X3_new))
                    if fitness_function(np.array([X3_new]), X, y)[0] < fitness_function(np.array([population[i]]), X, y)[0]:
                        population[i] = X3_new
                        if fitness_function(np.array([X3_new]), X, y)[0] < fitness_function(np.array([X_best]), X, y)[0]:
                            X_best = X3_new
                else:
                    # Narrowed exploitation (Step 41)
                    X4_new = population[i] + np.random.rand() * (X_best - population[i]) + np.random.rand() * (X_mean - population[i])
                    X4_new = convert_to_binary(apply_binary_transfer_function(X4_new))
                    if fitness_function(np.array([X4_new]), X, y)[0] < fitness_function(np.array([population[i]]), X, y)[0]:
                        population[i] = X4_new
                        if fitness_function(np.array([X4_new]), X, y)[0] < fitness_function(np.array([X_best]), X, y)[0]:
                            X_best = X4_new

            # Additional modification step based on random chance
            if rand2 <= 0.5:
                X_prime = binary_bhc(population[i], beta, bw)  # Apply binary BHC mutation
            else:
                X_prime = hho_mutation(population[i], X_best, population[np.random.randint(n)])  # Apply HHO mutation
                X_prime = convert_to_binary(apply_binary_transfer_function(X_prime))

            # Accept mutation if fitness improves
            if fitness_function(np.array([X_prime]), X, y)[0] < fitness_function(np.array([population[i]]), X, y)[0]:
                population[i] = X_prime

        # Increment iteration counter
        t += 1
    
    # Return the best solution found and its fitness value
    return X_best, fitness_function(X_best)

best_solution = ebao(n=30, T=100)
print("Best solution found:", best_solution)
