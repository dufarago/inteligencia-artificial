import heapq
import time
import os
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

class Grafo:
    def __init__(self, arquivo):
        """
        Inicializa o grafo a partir de um arquivo (Excel ou CSV)
        """
        self.vertices, self.arestas = self.ler_grafo(arquivo)
        self.heuristica = {}
    
    def ler_grafo(self, arquivo):
        """
        Lê o arquivo com tratamento robusto para diferentes formatos
        """
        try:
            # Verifica se o arquivo existe
            if not os.path.exists(arquivo):
                raise FileNotFoundError(f"Arquivo {arquivo} não encontrado")
            
            # Tenta ler como Excel
            try:
                df = pd.read_excel(arquivo, engine='openpyxl')
            except:
                # Se falhar, tenta como CSV
                try:
                    df = pd.read_csv(arquivo)
                except Exception as e:
                    raise ValueError(f"Não foi possível ler como Excel ou CSV: {str(e)}")
            
            # Verifica colunas obrigatórias
            colunas_necessarias = ['No_Origem', 'No_Destino', 'Peso']
            if not all(col in df.columns for col in colunas_necessarias):
                raise ValueError(f"O arquivo deve conter as colunas: {', '.join(colunas_necessarias)}")
            
            vertices = set()
            arestas = {}
            
            for _, row in df.iterrows():
                origem = str(row['No_Origem']).strip()
                destino = str(row['No_Destino']).strip()
                peso = float(row['Peso'])
                
                vertices.add(origem)
                vertices.add(destino)
                
                if origem not in arestas:
                    arestas[origem] = []
                arestas[origem].append((destino, peso))
            
            return vertices, arestas
            
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo}: {str(e)}")
            return set(), {}
            
        except FileNotFoundError:
            print(f"Erro: Arquivo {arquivo} não encontrado!")
            return set(), {}
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {str(e)}")
            return set(), {}

    def definir_heuristica(self, destino):
        
        for vertice in self.vertices:
            self.heuristica[vertice] = 1 if vertice != destino else 0

def medir_desempenho(algoritmo, grafo, inicio, fim):
    """
    Mede tempo de execução e retorna resultados
    """
    inicio_tempo = time.time()
    caminho, custo, nos_expandidos = algoritmo(grafo, inicio, fim)
    tempo_execucao = time.time() - inicio_tempo
    
    return {
        'caminho': caminho if caminho else [],
        'custo': custo if caminho else float('inf'),
        'tempo': tempo_execucao,
        'nos_expandidos': nos_expandidos
    }

# Algoritmos de Busca Cega
def bfs(grafo, inicio, fim):
    fila = deque([(inicio, [inicio], 0)])
    visitados = set()
    nos_expandidos = 0
    
    while fila:
        no, caminho, custo = fila.popleft()
        nos_expandidos += 1
        
        if no == fim:
            return caminho, custo, nos_expandidos
        
        if no not in visitados:
            visitados.add(no)
            for vizinho, peso in grafo.arestas.get(no, []):
                if vizinho not in visitados:
                    fila.append((vizinho, caminho + [vizinho], custo + peso))
    
    return None, float('inf'), nos_expandidos

def dfs(grafo, inicio, fim, limite=100):
    pilha = [(inicio, [inicio], 0)]
    visitados = set()
    nos_expandidos = 0
    
    while pilha:
        no, caminho, custo = pilha.pop()
        nos_expandidos += 1
        
        if no == fim:
            return caminho, custo, nos_expandidos
        
        if no not in visitados and len(caminho) < limite:
            visitados.add(no)
            for vizinho, peso in reversed(grafo.arestas.get(no, [])):
                if vizinho not in visitados:
                    pilha.append((vizinho, caminho + [vizinho], custo + peso))
    
    return None, float('inf'), nos_expandidos

# Algoritmos de Busca Heurística
def busca_gulosa(grafo, inicio, fim):
    grafo.definir_heuristica(fim)
    heap = [(grafo.heuristica[inicio], inicio, [inicio], 0)]
    visitados = set()
    nos_expandidos = 0
    
    while heap:
        _, no, caminho, custo = heapq.heappop(heap)
        nos_expandidos += 1
        
        if no == fim:
            return caminho, custo, nos_expandidos
        
        if no not in visitados:
            visitados.add(no)
            for vizinho, peso in grafo.arestas.get(no, []):
                heapq.heappush(heap, (grafo.heuristica[vizinho], vizinho, caminho + [vizinho], custo + peso))
    
    return None, float('inf'), nos_expandidos

def a_estrela(grafo, inicio, fim):
    grafo.definir_heuristica(fim)
    heap = [(0 + grafo.heuristica[inicio], inicio, [inicio], 0)]
    visitados = set()
    nos_expandidos = 0
    
    while heap:
        f, no, caminho, custo = heapq.heappop(heap)
        nos_expandidos += 1
        
        if no == fim:
            return caminho, custo, nos_expandidos
        
        if no not in visitados:
            visitados.add(no)
            for vizinho, peso in grafo.arestas.get(no, []):
                novo_custo = custo + peso
                heapq.heappush(heap, (novo_custo + grafo.heuristica[vizinho], vizinho, caminho + [vizinho], novo_custo))
    
    return None, float('inf'), nos_expandidos

def executar_testes():
    resultados = []
    
    for i in range(1, 11):
        arquivo = f'grafos/grafo_{i}.xlsx'
        
        if not os.path.exists(arquivo):
            print(f"⚠️ Arquivo {arquivo} não encontrado! Pulando...")
            continue
            
        try:
            print(f"\n🔍 Processando {arquivo}...")
            grafo = Grafo(arquivo)
            
            if not grafo.arestas:
                print(f"⚠️ Grafo {i} vazio ou inválido! Pulando...")
                continue
                
            nos = sorted(grafo.vertices, key=lambda x: int(x))  # Ordena nós numericamente
            inicio, fim = nos[0], nos[-1]
            print(f"  Nó inicial: {inicio}, Nó final: {fim}")
            
            algoritmos = [
                ('BFS', bfs),
                ('DFS', dfs),
                ('Gulosa', busca_gulosa),
                ('A*', a_estrela)
            ]
            
            for nome, algoritmo in algoritmos:
                print(f"  Executando {nome}...", end=' ')
                resultado = medir_desempenho(algoritmo, grafo, inicio, fim)
                status = "✅" if resultado['caminho'] else "❌"
                print(f"{status} Custo: {resultado['custo']}, Nós: {resultado['nos_expandidos']}")
                
                resultados.append({
                    'Grafo': f'grafo_{i}',
                    'Algoritmo': nome,
                    'Tempo (s)': round(resultado['tempo'], 4),
                    'Custo': resultado['custo'],
                    'Nós Expandidos': resultado['nos_expandidos'],
                    'Caminho': '→'.join(resultado['caminho']) if resultado['caminho'] else 'N/A'
                })
                
        except Exception as e:
            print(f"❌ Erro no grafo {i}: {str(e)}")
    
    return pd.DataFrame(resultados)

def gerar_relatorio(df):
    if df.empty:
        print("❌ Nenhum dado para gerar relatório!")
        return
    
    print("\n📊 Gerando relatório...")
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("Comparação de Algoritmos de Busca")
    
    # Tempo de Execução
    plt.subplot(2, 2, 1)
    df.groupby('Algoritmo')['Tempo (s)'].mean().plot.bar(color='skyblue')
    plt.title("Tempo Médio de Execução (s)")
    plt.ylabel("Segundos")
    
    # Custo do Caminho
    plt.subplot(2, 2, 2)
    df.groupby('Algoritmo')['Custo'].mean().plot.bar(color='lightgreen')
    plt.title("Custo Médio do Caminho")
    plt.ylabel("Soma dos Pesos")
    
    # Nós Expandidos
    plt.subplot(2, 2, 3)
    df.groupby('Algoritmo')['Nós Expandidos'].mean().plot.bar(color='salmon')
    plt.title("Nós Expandidos (Média)")
    plt.ylabel("Quantidade")
    
    # Exemplo de Caminho
    plt.subplot(2, 2, 4)
    sample = df[df['Caminho'] != 'N/A'].iloc[0]
    plt.text(0.1, 0.5, f"Exemplo de caminho ({sample['Algoritmo']}):\n{sample['Caminho']}", 
             fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('resultados.png', dpi=300)
    df.to_csv('resultados.csv', index=False)
    
    print("✅ Relatório gerado com sucesso!")
    print("📁 Arquivos criados:")
    print("- resultados.png (Gráficos comparativos)")
    print("- resultados.csv (Dados completos)")

if __name__ == '__main__':
    print("🚀 Iniciando análise de algoritmos de busca...")
    print("📌 Formato esperado dos grafos: No_Origem, No_Destino, Peso\n")
    
    df = executar_testes()
    
    if not df.empty:
        print("\n📋 Resultados sumarizados:")
        print(df[['Grafo', 'Algoritmo', 'Custo', 'Tempo (s)', 'Nós Expandidos']].to_string(index=False))
        gerar_relatorio(df)
    else:
        print("❌ Nenhum resultado válido foi gerado. Verifique:")
        print("- Os arquivos estão na pasta 'grafos' (grafo_1.xlsx a grafo_10.xlsx)")
        print("- As colunas são exatamente: No_Origem, No_Destino, Peso")
        print("- Os arquivos estão no formato .xlsx (Excel)")