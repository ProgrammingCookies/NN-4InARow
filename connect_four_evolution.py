import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import deque

class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols))
        self.current_player = 1
        self.game_over = False
        return self.get_state()
    
    def get_state(self):
        return self.board.copy()
    
    def is_valid_move(self, col):
        return self.board[0][col] == 0
    
    def get_valid_moves(self):
        return [col for col in range(self.cols) if self.is_valid_move(col)]
    
    def make_move(self, col):
        if not self.is_valid_move(col):
            return False, self.get_state(), -10, True
        
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break
        
        if self._check_win(row, col):
            self.game_over = True
            return True, self.get_state(), 1, True
        
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            return True, self.get_state(), 0, True
        
        self.current_player = 3 - self.current_player
        return True, self.get_state(), 0, False
    
    def _check_win(self, row, col):
        def check_direction(dr, dc):
            count = 1
            for direction in [-1, 1]:
                r, c = row + dr * direction, col + dc * direction
                while (0 <= r < self.rows and 0 <= c < self.cols and 
                       self.board[r][c] == self.current_player):
                    count += 1
                    r, c = r + dr * direction, c + dc * direction
            return count >= 4
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        return any(check_direction(dr, dc) for dr, dc in directions)

class ConnectFourNN(nn.Module):
    def __init__(self):
        super(ConnectFourNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)
        self.fitness = 0
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 6 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Evolution:
    def __init__(self, population_size=50, mutation_rate=0.1, mutation_strength=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [ConnectFourNN() for _ in range(population_size)]
        
    def mutate_model(self, model):
        child = copy.deepcopy(model)
        for param in child.parameters():
            if random.random() < self.mutation_rate:
                mutation = torch.randn_like(param) * self.mutation_strength
                param.data += mutation
        return child
    
    def crossover(self, model1, model2):
        child = ConnectFourNN()
        for param_name, param in child.named_parameters():
            if random.random() < 0.5:
                param.data.copy_(dict(model1.named_parameters())[param_name].data)
            else:
                param.data.copy_(dict(model2.named_parameters())[param_name].data)
        return child
    
    def select_parent(self, population):
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def create_next_generation(self):
        new_population = []
        # Keep the best performing models
        elite_size = self.population_size // 10
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_size]
        new_population.extend(elite)
        
        # Create the rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(self.population)
            parent2 = self.select_parent(self.population)
            child = self.crossover(parent1, parent2)
            child = self.mutate_model(child)
            new_population.append(child)
        
        self.population = new_population

def play_game_between_models(model1, model2, env):
    state = env.reset()
    done = False
    
    while not done:
        current_model = model1 if env.current_player == 1 else model2
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            q_values = current_model(state_tensor)
            valid_moves = env.get_valid_moves()
            valid_q_values = q_values[0][valid_moves]
            move = valid_moves[torch.argmax(valid_q_values).item()]
        
        _, state, reward, done = env.make_move(move)
    
    if reward == 1:  # Someone won
        return 1 if env.current_player == 2 else -1
    return 0  # Draw

def evaluate_population(population, games_per_match=10):
    env = ConnectFour()
    
    # Reset fitness scores
    for model in population:
        model.fitness = 0
    
    # Have each model play against every other model
    for i, model1 in enumerate(population):
        for j, model2 in enumerate(population[i+1:], i+1):
            for _ in range(games_per_match):
                # Play one game as player 1 and one as player 2
                result1 = play_game_between_models(model1, model2, env)
                result2 = play_game_between_models(model2, model1, env)
                
                model1.fitness += result1 - result2
                model2.fitness += result2 - result1

def train_evolutionary(generations=100, population_size=50, games_per_match=10):
    evolution = Evolution(population_size=population_size)
    best_fitness_history = []
    
    for gen in range(generations):
        # Evaluate the current population
        evaluate_population(evolution.population, games_per_match)
        
        # Get statistics for this generation
        best_model = max(evolution.population, key=lambda x: x.fitness)
        avg_fitness = sum(model.fitness for model in evolution.population) / len(evolution.population)
        best_fitness_history.append(best_model.fitness)
        
        print(f"Generation {gen + 1}")
        print(f"Best Fitness: {best_model.fitness}")
        print(f"Average Fitness: {avg_fitness}")
        
        # Create the next generation
        evolution.create_next_generation()
    
    # Return the best model from the final generation
    return max(evolution.population, key=lambda x: x.fitness), best_fitness_history

def play_against_evolved_model(model, human_player=1):
    env = ConnectFour()
    state = env.reset()
    done = False
    
    while not done:
        print(np.flip(state, axis=0))
        
        if env.current_player == human_player:
            valid_moves = env.get_valid_moves()
            while True:
                try:
                    move = int(input(f"Enter your move (0-6): "))
                    if move in valid_moves:
                        break
                    print("Invalid move, try again")
                except ValueError:
                    print("Please enter a number between 0 and 6")
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
                valid_moves = env.get_valid_moves()
                valid_q_values = q_values[0][valid_moves]
                move = valid_moves[torch.argmax(valid_q_values).item()]
        
        _, state, reward, done = env.make_move(move)
        
        if done:
            print(np.flip(state, axis=0))
            if reward == 1:
                print(f"Player {3 - env.current_player} wins!")
            elif reward == 0:
                print("It's a draw!")
            else:
                print("Invalid move!")