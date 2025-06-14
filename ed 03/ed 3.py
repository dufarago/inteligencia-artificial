import pandas as pd
import random
import numpy as np
from collections import defaultdict

def carregar_instancia(nome_arquivo):
    try:
        df = pd.read_csv(nome_arquivo)  
        if 'Peso' not in df.columns or 'Valor' not in df.columns:
            raise ValueError("Arquivo CSV não contém as colunas esperadas 'Peso' e 'Valor'.")
        
        capacidade = int(df.iloc[-1, 1])
        df = df.drop(df.index[-1])
        itens = list(zip(df['Peso'], df['Valor']))
        
        print(f"\nCapacidade da mochila para {nome_arquivo}: {capacidade}")
        return itens, capacidade
    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {e}")
        return [], 0

def calcular_fitness(individuo, itens, capacidade):
    peso_total = 0
    valor_total = 0
    for i in range(len(individuo)):
        if individuo[i] == 1:
            peso_total += itens[i][0]
            valor_total += itens[i][1]
    return (0, 0) if peso_total > capacidade else (valor_total, peso_total)

def inicializar_populacao_aleatoria(tamanho_populacao, num_itens):
    return [[random.randint(0, 1) for _ in range(num_itens)] for _ in range(tamanho_populacao)]

def inicializar_populacao_heuristica(tamanho_populacao, itens, capacidade):
    populacao = []
    num_itens = len(itens)
    
    # Ordena itens por valor/peso (heurística gulosa)
    itens_ordenados = sorted([(i, itens[i][1]/itens[i][0]) for i in range(num_itens)], 
                            key=lambda x: x[1], reverse=True)
    
    for _ in range(tamanho_populacao):
        individuo = [0] * num_itens
        peso_atual = 0
        
        # Adiciona itens até atingir a capacidade
        for i, _ in itens_ordenados:
            if peso_atual + itens[i][0] <= capacidade and random.random() > 0.3:  # 70% chance de adicionar
                individuo[i] = 1
                peso_atual += itens[i][0]
        
        # Adiciona alguns itens aleatórios para diversidade
        for _ in range(int(num_itens * 0.1)):  # 10% dos itens
            idx = random.randint(0, num_itens-1)
            if individuo[idx] == 0 and peso_atual + itens[idx][0] <= capacidade:
                individuo[idx] = 1
                peso_atual += itens[idx][0]
        
        populacao.append(individuo)
    
    return populacao

def selecionar_roleta(populacao, fitness_populacao):
    soma_fitness = sum(f[0] for f in fitness_populacao)
    if soma_fitness == 0:
        return random.choices(populacao, k=2)
    probabilidades = [f[0]/soma_fitness for f in fitness_populacao]
    return random.choices(populacao, weights=probabilidades, k=2)

def selecionar_torneio(populacao, fitness_populacao, tamanho_torneio=3):
    selecionados = []
    for _ in range(2):
        competidores = random.sample(list(zip(populacao, fitness_populacao)), tamanho_torneio)
        vencedor = max(competidores, key=lambda x: x[1][0])
        selecionados.append(vencedor[0])
    return selecionados

def cruzar_um_ponto(pai, mae):
    ponto_corte = random.randint(1, len(pai)-1)
    filho1 = pai[:ponto_corte] + mae[ponto_corte:]
    filho2 = mae[:ponto_corte] + pai[ponto_corte:]
    return filho1, filho2

def cruzar_dois_pontos(pai, mae):
    ponto1 = random.randint(1, len(pai)-2)
    ponto2 = random.randint(ponto1, len(pai)-1)
    filho1 = pai[:ponto1] + mae[ponto1:ponto2] + pai[ponto2:]
    filho2 = mae[:ponto1] + pai[ponto1:ponto2] + mae[ponto2:]
    return filho1, filho2

def cruzar_uniforme(pai, mae, taxa_crossover=0.5):
    filho1 = []
    filho2 = []
    for i in range(len(pai)):
        if random.random() < taxa_crossover:
            filho1.append(mae[i])
            filho2.append(pai[i])
        else:
            filho1.append(pai[i])
            filho2.append(mae[i])
    return filho1, filho2

def mutar_binaria(individuo, taxa_mutacao):
    return [1-gene if random.random() < taxa_mutacao else gene for gene in individuo]

def mutar_troca(individuo, taxa_mutacao):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            j = random.randint(0, len(individuo)-1)
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

def algoritmo_genetico(itens, capacidade, config):
    num_itens = len(itens)
    
    # Configurações do algoritmo
    tamanho_populacao = config.get('tamanho_populacao', 50)
    geracoes = config.get('geracoes', 100)
    taxa_mutacao = config.get('taxa_mutacao', 0.1)
    taxa_crossover = config.get('taxa_crossover', 0.8)
    metodo_selecao = config.get('metodo_selecao', 'roleta')
    metodo_crossover = config.get('metodo_crossover', 'um_ponto')
    metodo_mutacao = config.get('metodo_mutacao', 'binaria')
    inicializacao = config.get('inicializacao', 'aleatoria')
    criterio_parada = config.get('criterio_parada', 'geracoes')
    limite_convergencia = config.get('limite_convergencia', 20)
    
    # Inicialização da população
    if inicializacao == 'aleatoria':
        populacao = inicializar_populacao_aleatoria(tamanho_populacao, num_itens)
    else:
        populacao = inicializar_populacao_heuristica(tamanho_populacao, itens, capacidade)
    
    historico_fitness = []
    melhor_solucao = None
    melhor_fitness = -1
    melhor_peso = 0
    contador_convergencia = 0
    
    print("\nConfigurações utilizadas:")
    print(f" - Tamanho da população: {tamanho_populacao}")
    print(f" - Taxa de crossover: {taxa_crossover}")
    print(f" - Taxa de mutação: {taxa_mutacao}")
    print(f" - Método de seleção: {metodo_selecao}")
    print(f" - Método de crossover: {metodo_crossover}")
    print(f" - Método de mutação: {metodo_mutacao}")
    print(f" - Inicialização: {inicializacao}")
    print(f" - Critério de parada: {criterio_parada}")
    if criterio_parada == 'convergencia':
        print(f" - Limite de convergência: {limite_convergencia} gerações sem melhora")
    
    print("\nEvolução do fitness por geração:")
    print("Geração | Melhor Fitness | Peso | Média Fitness")
    print("-----------------------------------------------")
    
    for geracao in range(geracoes):
        fitness_populacao = [calcular_fitness(ind, itens, capacidade) for ind in populacao]
        media_fitness = sum(f[0] for f in fitness_populacao) / len(fitness_populacao)
        
        # Encontrar a melhor solução atual
        melhor_atual = max(fitness_populacao, key=lambda x: x[0])
        if melhor_atual[0] > melhor_fitness:
            melhor_fitness, melhor_peso = melhor_atual
            melhor_solucao = populacao[fitness_populacao.index(melhor_atual)]
            contador_convergencia = 0
        else:
            contador_convergencia += 1
        
        # Registrar histórico
        historico_fitness.append((melhor_fitness, media_fitness))
        
        # Mostrar progresso
        print(f"{geracao:6} | {melhor_fitness:14} | {melhor_peso:4} | {media_fitness:12.2f}")
        
        # Verificar critério de parada por convergência
        if criterio_parada == 'convergencia' and contador_convergencia >= limite_convergencia:
            print(f"\nConvergência atingida após {geracao} gerações sem melhora.")
            break
        
        # Nova população
        nova_populacao = []
        for _ in range(tamanho_populacao // 2):
            # Seleção
            if metodo_selecao == 'roleta':
                pai, mae = selecionar_roleta(populacao, fitness_populacao)
            else:
                pai, mae = selecionar_torneio(populacao, fitness_populacao)
            
            # Crossover
            if random.random() < taxa_crossover:
                if metodo_crossover == 'um_ponto':
                    filho1, filho2 = cruzar_um_ponto(pai, mae)
                elif metodo_crossover == 'dois_pontos':
                    filho1, filho2 = cruzar_dois_pontos(pai, mae)
                else:
                    filho1, filho2 = cruzar_uniforme(pai, mae)
            else:
                filho1, filho2 = pai.copy(), mae.copy()
            
            # Mutação
            if metodo_mutacao == 'binaria':
                filho1 = mutar_binaria(filho1, taxa_mutacao)
                filho2 = mutar_binaria(filho2, taxa_mutacao)
            else:
                filho1 = mutar_troca(filho1, taxa_mutacao)
                filho2 = mutar_troca(filho2, taxa_mutacao)
            
            nova_populacao.extend([filho1, filho2])
        
        # Elitismo: manter a melhor solução
        if melhor_solucao:
            nova_populacao[0] = melhor_solucao
        
        populacao = nova_populacao
    
    # Resultado final
    print("\nMelhor solução encontrada:")
    print(f"Valor total: {melhor_fitness}")
    print(f"Peso total: {melhor_peso}")
    print(f"Capacidade: {capacidade}")
    print(f"Itens selecionados: {[i for i in range(num_itens) if melhor_solucao[i] == 1]}")
    
    return melhor_solucao, historico_fitness

def testar_configuracoes(itens, capacidade):
    configuracoes = [
        # Testes de crossover
        {'metodo_crossover': 'um_ponto', 'descricao': "Crossover - Um Ponto"},
        {'metodo_crossover': 'dois_pontos', 'descricao': "Crossover - Dois Pontos"},
        {'metodo_crossover': 'uniforme', 'descricao': "Crossover - Uniforme"},
        
        # Testes de mutação
        {'metodo_mutacao': 'binaria', 'taxa_mutacao': 0.01, 'descricao': "Mutação - Baixa (1%)"},
        {'metodo_mutacao': 'binaria', 'taxa_mutacao': 0.1, 'descricao': "Mutação - Média (10%)"},
        {'metodo_mutacao': 'binaria', 'taxa_mutacao': 0.3, 'descricao': "Mutação - Alta (30%)"},
        
        # Testes de inicialização
        {'inicializacao': 'aleatoria', 'descricao': "Inicialização - Aleatória"},
        {'inicializacao': 'heuristica', 'descricao': "Inicialização - Heurística"},
        
        # Testes de critério de parada
        {'criterio_parada': 'geracoes', 'descricao': "Parada - Gerações Fixas"},
        {'criterio_parada': 'convergencia', 'limite_convergencia': 10, 'descricao': "Parada - Convergência (10 gerações)"},
    ]
    
    resultados = []
    
    for config in configuracoes:
        print(f"\n=== TESTANDO CONFIGURAÇÃO: {config['descricao']} ===")
        
        # Configurações padrão
        config_padrao = {
            'tamanho_populacao': 50,
            'geracoes': 100,
            'taxa_mutacao': 0.1,
            'taxa_crossover': 0.8,
            'metodo_selecao': 'roleta',
            'metodo_crossover': 'um_ponto',
            'metodo_mutacao': 'binaria',
            'inicializacao': 'aleatoria',
            'criterio_parada': 'geracoes'
        }
        
        # Atualiza com a configuração atual
        config_padrao.update(config)
        
        solucao, historico = algoritmo_genetico(itens, capacidade, config_padrao)
        melhor_fitness = max(h[0] for h in historico)
        resultados.append((config['descricao'], melhor_fitness, historico[-1][1]))
    
    # Exibir resultados comparativos
    print("\n=== RESULTADOS COMPARATIVOS ===")
    print("Configuração | Melhor Fitness | Média Final")
    print("-------------------------------------------")
    for descricao, melhor, media in resultados:
        print(f"{descricao:30} | {melhor:14} | {media:11.2f}")
    
    return resultados

def executar_todos_arquivos():
    resultados_gerais = defaultdict(list)
    
    for i in range(1, 11):
        arquivo = f'instancias/knapsack_{i}.csv'
        print(f"\n=== PROCESSANDO ARQUIVO: {arquivo} ===")
        
        try:
            itens, capacidade = carregar_instancia(arquivo)
            if itens:
                resultados = testar_configuracoes(itens, capacidade)
                
                # Armazenar resultados para análise geral
                for descricao, melhor, media in resultados:
                    resultados_gerais[descricao].append(melhor)
            else:
                print(f"Arquivo {arquivo} vazio ou inválido.")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
    
    # Exibir análise geral de todas as instâncias
    print("\n=== ANÁLISE GERAL DE TODAS AS INSTÂNCIAS ===")
    print("Configuração | Média Melhor Fitness")
    print("----------------------------------")
    for descricao, valores in resultados_gerais.items():
        media = sum(valores) / len(valores)
        print(f"{descricao:30} | {media:18.2f}")

executar_todos_arquivos()