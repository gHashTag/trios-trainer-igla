# Queen-Hive MCP

L-SS3 (`#156`) · Epic [gHashTag/trios#940](https://github.com/gHashTag/trios/issues/940)

**Единственный sovereign control plane** для флота Sovereign Scarab v4. Пишет в `ssot.scarab_strategy` через 7 MCP-инструментов. Никакого Railway API, PAT-токенов или `variableUpsert`.

## Запуск

```bash
DATABASE_URL=postgres://… cargo run --release --bin queen-hive-mcp
```

Сервер слушает JSON-RPC 2.0 на stdin/stdout, логирует на stderr.

## Tools

| Tool | DB function | Confirm? |
|---|---|---|
| `spawn_scarab` | `ssot.spawn_scarab(optimizer, format, hidden, lr, seed, steps, service_id)` | — |
| `bump_strategy` | `ssot.bump_strategy_v2(canon_name, …)` | — |
| `pause_scarab` | `ssot.pause_scarab(canon_name)` | — |
| `resume_scarab` | `ssot.resume_scarab(canon_name)` | — |
| `kill_scarab` | `ssot.kill_scarab(canon_name)` | **YES (R9)** |
| `fleet_status` | `SELECT * FROM ssot.fleet_status` | — |
| `emergency_mass_op` | bulk pause/resume/kill | **YES (R9 + blast guard >5)** |

## Защитные инварианты

- **R-SI-1** zero star operator
- **R9** `confirm=true` для деструктивных операций
- **Blast-radius guard** mass-op на >5 узлов требует `confirm=true`
- **NUMERIC-STANDARD-001** seeds Fibonacci whitelist `{1597, 2584, 4181, 6765, 10946, 47, 89, 144, 123}` валидируется до записи в БД (CHECK constraint в SSOT тоже)
- **Compile-time sentinel test** запрещает любое упоминание `RAILWAY_TOKEN`, `variableUpsert`, `railway.app/graphql` в исходниках

## Acceptance — G-SS-06 + G-SS-07 + G-SS-08

```bash
rg -n 'RAILWAY_TOKEN|variableUpsert' queen-hive-mcp/   # → пусто
cargo test --release                                    # → 5/5 pass
```

## MCP client config (пример `.mcp.json`)

```json
{
  "mcpServers": {
    "queen-hive": {
      "command": "/path/to/queen-hive-mcp",
      "env": { "DATABASE_URL": "postgres://…" }
    }
  }
}
```

Anchor: φ² + φ⁻² = 3 · DOI [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)
