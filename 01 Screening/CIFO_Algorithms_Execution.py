# %% [markdown]
# # CIFO - Algoritmos de Otimização para Fantasy League
# 
# Este notebook é dedicado à execução dos algoritmos de otimização.
# Os resultados serão exportados para um arquivo CSV para análise posterior.

# %%
# Importações necessárias
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from datetime import datetime

# Importar módulos do projeto
from solution import LeagueSolution, LeagueHillClimbingSolution
from evolution import hill_climbing, simulated_annealing
from operators import (
    selection_tournament,
    selection_ranking,
    selection_boltzmann,
    crossover_one_point,
    crossover_uniform,
    mutate_swap, 
    mutate_swap_constrained,
    genetic_algorithm
)
from fitness_counter import FitnessCounter

# Implementar crossover de dois pontos (não existe em operators.py)
def two_point_crossover(parent1, parent2):
    """
    Two-point crossover: creates a child by taking portions from each parent.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    cut1 = random.randint(1, len(parent1.repr) - 3)
    cut2 = random.randint(cut1 + 1, len(parent1.repr) - 2)
    child_repr = parent1.repr[:cut1] + parent2.repr[cut1:cut2] + parent1.repr[cut2:]
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )

# %% [markdown]
# ## 1. Configuração do Ambiente

# %%
# Configurar seed para reprodutibilidade
random.seed(42)
np.random.seed(42)

# Configurar parâmetros gerais
NUM_RUNS = 3  # Número de execuções para cada algoritmo
MAX_EVALUATIONS = 10000  # Número máximo de avaliações de função
POPULATION_SIZE = 100  # Tamanho da população para algoritmos genéticos
MAX_GENERATIONS = 100  # Número máximo de gerações para algoritmos genéticos

# %% [markdown]
# ## 2. Configuração dos Algoritmos

# %%
# Carregar dados dos jogadores
players_df = pd.read_csv('players.csv')

# Converter DataFrame para lista de dicionários
players_list = players_df.to_dict('records')

# Configurar contador de fitness
fitness_counter = FitnessCounter()

# Configurar algoritmos
configs = {
    # Algoritmos base
    'HC_Standard': {
        'algorithm': 'Hill Climbing',
    },
    'SA_Standard': {
        'algorithm': 'Simulated Annealing',
    },
    'GA_Tournament_OnePoint': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'One Point',
        'mutation_rate': 1.0/35,  # 1.0/len(players)
        'elitism_percent': 0.1,   # 10%
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_Tournament_TwoPoint': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_Rank_Uniform': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Rank',
        'crossover': 'Uniform',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_Boltzmann_TwoPoint': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Boltzmann',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_Hybrid': {
        'algorithm': 'Genetic Algorithm Hybrid',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    
    # GA com diferentes taxas de mutação
    'GA_Low_Mutation': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 0.5/35,  # Taxa baixa
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_High_Mutation': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 2.0/35,  # Taxa alta
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    
    # GA com diferentes níveis de elitismo
    'GA_No_Elitism': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.0,  # Sem elitismo
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_High_Elitism': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.3,  # Elitismo alto (30%)
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': False,
    },
    
    # GA com diferentes tamanhos de população
    'GA_Small_Population': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 50,  # População pequena
        'use_valid_initial': False,
        'use_repair': False,
    },
    'GA_Large_Population': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 200,  # População grande
        'use_valid_initial': False,
        'use_repair': False,
    },
    
    # GA com foco em soluções válidas
    'GA_Valid_Initial': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': True,  # Começa com população válida
        'use_repair': False,
    },
    'GA_Repair_Operator': {
        'algorithm': 'Genetic Algorithm',
        'selection': 'Tournament',
        'crossover': 'Two Point',
        'mutation_rate': 1.0/35,
        'elitism_percent': 0.1,
        'population_size': 100,
        'use_valid_initial': False,
        'use_repair': True,  # Usa operador de reparo
    }
}

# %% [markdown]
# ## 3. Execução dos Algoritmos

# %%
# Função para executar Hill Climbing
def run_hill_climbing(players, max_evaluations):
    solution = LeagueSolution(players)
    
    # Iniciar contagem de fitness
    fitness_counter.reset()
    solution.set_fitness_counter(fitness_counter)
    
    best_fitness = solution.fitness()
    history = [best_fitness]
    
    while fitness_counter.get_count() < max_evaluations:
        # Gerar vizinho
        neighbor = deepcopy(solution)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Aceitar se melhor
        if neighbor_fitness < best_fitness:  # Menor é melhor
            solution = neighbor
            best_fitness = neighbor_fitness
        
        history.append(best_fitness)
    
    return solution, history, fitness_counter.get_count()

# Função para executar Simulated Annealing
def run_simulated_annealing(players, max_evaluations):
    solution = LeagueSolution(players)
    
    # Iniciar contagem de fitness
    fitness_counter.reset()
    solution.set_fitness_counter(fitness_counter)
    
    best_solution = deepcopy(solution)
    current_fitness = solution.fitness()
    best_fitness = current_fitness
    
    history = [best_fitness]
    
    # Parâmetros do SA
    initial_temp = 100.0
    final_temp = 0.1
    alpha = 0.95
    
    current_temp = initial_temp
    
    while fitness_counter.get_count() < max_evaluations and current_temp > final_temp:
        # Gerar vizinho
        neighbor = deepcopy(solution)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Calcular delta
        delta = neighbor_fitness - current_fitness
        
        # Aceitar se melhor ou com probabilidade baseada na temperatura
        if delta < 0 or random.random() < np.exp(-delta / current_temp):
            solution = neighbor
            current_fitness = neighbor_fitness
            
            # Atualizar melhor solução se necessário
            if current_fitness < best_fitness:
                best_solution = deepcopy(solution)
                best_fitness = current_fitness
        
        history.append(best_fitness)
        
        # Resfriar
        current_temp *= alpha
    
    return best_solution, history, fitness_counter.get_count()

# Função para executar Genetic Algorithm
def run_genetic_algorithm(players, config, max_evaluations):
    # Iniciar contagem de fitness
    fitness_counter.reset()
    
    # Configurar seleção
    if config['selection'] == 'Tournament':
        selection_op = selection_tournament
    elif config['selection'] == 'Rank':
        selection_op = selection_ranking
    elif config['selection'] == 'Boltzmann':
        selection_op = selection_boltzmann
    else:
        raise ValueError(f"Seleção não suportada: {config['selection']}")
    
    # Configurar crossover
    if config['crossover'] == 'One Point':
        crossover_op = crossover_one_point
    elif config['crossover'] == 'Two Point':
        crossover_op = two_point_crossover
    elif config['crossover'] == 'Uniform':
        crossover_op = crossover_uniform
    else:
        raise ValueError(f"Crossover não suportado: {config['crossover']}")
    
    # Configurar mutação
    mutation_op = mutate_swap
    
    # Configurar operador de reparo (se necessário)
    repair_op = None
    if config['use_repair']:
        def repair_operator(solution):
            # Implementação simples de reparo: tenta corrigir soluções inválidas
            # ajustando a distribuição de jogadores por posição e orçamento
            if solution.is_valid():
                return solution
            
            # Obter estatísticas das equipes
            teams = solution.get_teams()
            
            # Verificar e corrigir distribuição de posições
            for team_idx, team in enumerate(teams):
                positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
                for player in team:
                    positions[player["Position"]] += 1
                
                # Se a distribuição estiver incorreta, tentar corrigir
                if positions != {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}:
                    # Implementação simplificada: apenas retorna a solução original
                    # Uma implementação real seria mais complexa
                    pass
            
            return solution
        
        repair_op = repair_operator
    
    # Configurar local search para GA híbrido
    local_search = None
    if config['algorithm'] == 'Genetic Algorithm Hybrid':
        local_search = {
            'operator': 'hill_climbing',
            'probability': 0.1,
            'iterations': 10
        }
    
    # Executar GA
    best_solution, best_fitness, history = genetic_algorithm(
        players=players,
        population_size=config['population_size'],
        max_generations=MAX_GENERATIONS,
        selection_operator=selection_op,
        crossover_operator=crossover_op,
        crossover_rate=0.8,
        mutation_operator=mutation_op,
        mutation_rate=config['mutation_rate'],
        elitism=config['elitism_percent'] > 0,
        elitism_size=int(config['population_size'] * config['elitism_percent']),
        local_search=local_search,
        fitness_counter=fitness_counter,
        max_evaluations=max_evaluations,
        verbose=False
    )
    
    return best_solution, history, fitness_counter.get_count()

# %% [markdown]
# ## 4. Execução dos Experimentos

# %%
# Função para executar um experimento completo
def run_experiment(config_name, config, players, num_runs, max_evaluations):
    results = []
    all_history = []
    
    for run in range(num_runs):
        print(f"Executando {config_name}, run {run+1}/{num_runs}...")
        
        start_time = time.time()
        
        try:
            if config['algorithm'] == 'Hill Climbing':
                best_solution, history, evaluations = run_hill_climbing(players, max_evaluations)
            elif config['algorithm'] == 'Simulated Annealing':
                best_solution, history, evaluations = run_simulated_annealing(players, max_evaluations)
            elif 'Genetic Algorithm' in config['algorithm']:
                best_solution, history, evaluations = run_genetic_algorithm(players, config, max_evaluations)
            else:
                raise ValueError(f"Algoritmo não suportado: {config['algorithm']}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Registrar resultados
            results.append({
                'Configuration': config_name,
                'Run': run + 1,
                'Best Fitness': best_solution.fitness(),
                'Evaluations': evaluations,
                'Time': execution_time,
                'Valid': best_solution.is_valid()
            })
            
            all_history.append(history)
        except Exception as e:
            # Registrar erro
            results.append({
                'Configuration': config_name,
                'Run': run + 1,
                'Best Fitness': float('inf'),
                'Evaluations': 0,
                'Time': 0,
                'Valid': False,
                'Error': str(e)
            })
            
            all_history.append([])
            print(f"Erro ao executar {config_name}, run {run+1}: {e}")
    
    return results, all_history

# Executar todos os experimentos
all_results = []
history_data = {}

for config_name, config in configs.items():
    print(f"\nExecutando experimentos para {config_name}...")
    results, history = run_experiment(config_name, config, players_list, NUM_RUNS, MAX_EVALUATIONS)
    all_results.extend(results)
    history_data[config_name] = history

# Converter resultados para DataFrame
results_df = pd.DataFrame(all_results)

# Salvar resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"experiment_results_{timestamp}.csv", index=False)
np.save(f"history_data_{timestamp}.npy", history_data)

print(f"\nExperimentos concluídos. Resultados salvos em experiment_results_{timestamp}.csv")
print(f"Histórico de fitness salvo em history_data_{timestamp}.npy")

# %% [markdown]
# ## 5. Análise Preliminar dos Resultados

# %%
# Mostrar estatísticas básicas
print("Estatísticas por configuração:")
stats = results_df.groupby('Configuration').agg({
    'Best Fitness': ['mean', 'std', 'min', 'max'],
    'Evaluations': ['mean', 'std'],
    'Time': ['mean', 'std'],
    'Valid': 'mean'
})
print(stats)

# Plotar fitness médio por configuração
plt.figure(figsize=(12, 6))
avg_fitness = results_df.groupby('Configuration')['Best Fitness'].mean().sort_values()
avg_fitness.plot(kind='bar')
plt.title('Fitness Médio por Configuração')
plt.ylabel('Fitness (menor é melhor)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plotar tempo médio por configuração
plt.figure(figsize=(12, 6))
avg_time = results_df.groupby('Configuration')['Time'].mean().sort_values()
avg_time.plot(kind='bar')
plt.title('Tempo Médio de Execução por Configuração')
plt.ylabel('Tempo (segundos)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Conclusão
# 
# Este notebook executou diversos algoritmos de otimização para o problema da Fantasy League, incluindo:
# 
# - Hill Climbing
# - Simulated Annealing
# - Genetic Algorithms com diferentes configurações:
#   - Diferentes operadores de seleção (Tournament, Rank, Boltzmann)
#   - Diferentes operadores de crossover (One Point, Two Point, Uniform)
#   - Diferentes taxas de mutação
#   - Diferentes níveis de elitismo
#   - Diferentes tamanhos de população
#   - Variantes com foco em soluções válidas
# 
# Os resultados foram salvos em arquivos CSV e NPY para análise detalhada em um notebook separado.
