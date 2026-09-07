# Contrato de dados analíticos

## Responsabilidade e consumidores

- **Owner:** projeto Lake FastF1.
- **Origem:** resultados FastF1 consolidados em `bronze/results`.
- **Consumidores:** Streamlit, espelho MySQL e análises de BI.
- **Atualização:** semanal, após a ingestão do resultado mais recente.

## `mart_driver_round`

- **Grão:** um piloto em uma corrida de uma temporada.
- **Chave:** `(season, round_number, driver_id)`.
- **Tempo de evento:** `event_date`; a data não representa o instante de ingestão.
- `official_grid` é nulo para grid zero/pit lane.
- `official_finish` existe apenas para classificação numérica.
- `result_status` pertence a `FINISHED`, `DNF`, `DNS`, `DNQ`, `DSQ` ou `NC`.
- `started` é falso para DNS e DNQ.

## `mart_standings`

- **Grão:** um piloto em cada snapshot de rodada da temporada.
- **Chave:** `(season, round_number, driver_id)`.
- `round_points` soma Race e Sprint; ausência em uma rodada gera zero e preserva `cumulative_points`.
- `championship_rank` usa pontos acumulados e mantém empates com o mesmo rank.
- Nome, equipe e cor representam os metadados mais recentes disponíveis na temporada.

## Qualidade e operação

- Chaves devem ser únicas e campos de chave não podem ser nulos.
- Pontos e posições não podem ser negativos; probabilidades ficam em `[0, 1]`.
- Probabilidades normalizadas devem somar 1 por snapshot, dentro de tolerância `1e-9`.
- O dashboard exibe a última data disponível e a contagem de duplicidades.
- Escritas Silver são idempotentes por overwrite; um rerun recompõe todo o histórico.
- Backfills começam na Raw e recompõem Bronze, Silver e espelho MySQL na ordem do DAG.
- Mudanças incompatíveis exigem novo mart ou versão de API; os marts atuais não mudam de grão silenciosamente.

## Limitações

- Os marts cobrem resultados; não incluem telemetria, clima, pneus ou voltas.
- Comparações de pontos entre eras exigem contexto regulatório e não são tratadas como equivalentes.
