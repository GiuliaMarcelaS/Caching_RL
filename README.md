Os 4 modos (`qlearn`, `lru`, `oracle`, `nocache`) do notebook original foram
mantidos como baselines de comparacao, rodam automaticamente junto do
`Optimal_QLRU` em cada execucao (ver `scripts/_common.py`).

## Trafego real (traffic_real.py)
popularidade empirica + amostragem i.i.d,
nao a sequencia temporal bruta. Ou seja, `traffic_real.py` conta quantas
vezes cada funcao foi invocada no trace real, usa isso como `popularities`
(mesmo formato/contrato de `req_generator.zipf`/`Uniform`), e o resto do
pipeline nao muda -- `np.random.choice(F, T, p=popularities)` continua
igual ao que o notebook ja faz.


O script informa quantas funcoes distintas existem -- precisa ser >=
`catalog_size` (100.000 na escala cheia).

## Rodando os experimentos

```bash

# Experimento 1 (escala: F=400, T=10k, capacity=1% do catalogo)
python scripts/run_experiment1.py --traffic both
python scripts/summarize_results.py results/experiment1_scalability.csv --group-col num_servers

# Experimento 2 (troque 100 pela melhor qtd de servidores do Exp. 1)
python scripts/run_experiment2.py --traffic both --num-servers 100
python scripts/summarize_results.py results/experiment2_capacity.csv --group-col capacity
```

Cada execucao roda **5 algoritmos** por linha de configuracao:
`Optimal_QLRU` (cache.py) + `qlearn`/`lru`/`oracle`/`nocache`
(qlearning_balancer.py) -- todos usando o mesmo `Monitor`/popularidade/
requisicoes daquela execucao, entao sao diretamente comparaveis linha a
linha no CSV.

## Desempenho medido -- leia antes de rodar em escala cheia
Para a escala cheia (F=100.000, T=10.000.000, capacity=1% do catalogo, 30 execucoes por configuracao), a estimativa de tempo de CPU para o Experimento 1 é ~376h-core (ou seja, 15 dias em 1 core). Para o Experimento 2, a estimativa é ~188h-core (ou seja, 7 dias em 1 core). Entao podemos reduzir o custo de CPU e tempo real com algumas estrategias:

1. **Rode um piloto primeiro**: `--num-runs 5` (em vez de 30) só para achar
   o "melhor num_servers" do Experimento 1 rapido, e só depois rode as 30
   execucoes completas
2. **Use uma maquina/cluster com muitos cores** e `--workers` = total de
   cores. Em 32 cores, a estimativa de ~376h-core cai para ~12h "reais".
3. O `--num-requests` (T) por padrão do é 10.000 para o experimento nao durar tanto tempo. Mas 
    pode aumentar para a escala cheia de 10.000.000 como no notebook
4. Os modos `lru`/`oracle`/`nocache` nao tem aleatoriedade de exploracao
   (so variam pela amostragem do catalogo/trafego a cada seed) -- se
   precisar cortar custo, eles sao os primeiros candidatos a rodar com
   menos execucoes que o `qlearn` (que precisa de mais amostras para a
   tabela Q convergir de forma estavel).

