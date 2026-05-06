# rainfrog runbook — IGLA RACE operator console

> Anchor: `φ² + φ⁻² = 3` · DOI [10.5281/zenodo.19227877](https://doi.org/10.5281/zenodo.19227877)
> ONE SHOT [trios-trainer-igla#99 RAINFROG-ADOPT](https://github.com/gHashTag/trios-trainer-igla/issues/99) · R1 Rust-only

`igla-dash` runs two databases:

| Leg                | Driver   | Path / URL                                               | Lane    |
| ------------------ | -------- | -------------------------------------------------------- | ------- |
| `igla-sqlite`      | sqlite   | `./data.db` (local) or `/data/igla.db` (Railway)         | default |
| `phd-ssot`         | postgres | `phd-postgres-ssot` Railway TCP (host:port via L-RF4)    | optional |

We use [rainfrog](https://github.com/achristmascarl/rainfrog) — a vim-flavoured
ratatui+sqlx TUI — as the only operator console.  No web UI, no Drizzle Studio.

---

## Install

```bash
cargo install --locked rainfrog
```

Verify:

```bash
rainfrog --help    # should print usage with --url / --host / --driver
```

---

## Configure

Copy the template the first time:

```bash
mkdir -p ~/.config/rainfrog
cp etc/rainfrog.config.toml.example ~/.config/rainfrog/rainfrog_config.toml
```

`rainfrog_config.toml` location follows the [`directories`](https://docs.rs/directories) crate:

| Platform | Path                                                  |
| -------- | ----------------------------------------------------- |
| Linux    | `$XDG_CONFIG_HOME/rainfrog/rainfrog_config.toml`      |
| macOS    | `~/Library/Application Support/rainfrog/rainfrog_config.toml` |
| Windows  | `%LOCALAPPDATA%\rainfrog\config\rainfrog_config.toml` |

Override with `RAINFROG_CONFIG=~/.config/rainfrog`.

Connection priority (rainfrog docs): `--url` flag > `$DATABASE_URL` > config file.

---

## Connect

```bash
./scripts/dash            # SQLite leg (default; uses ./data.db or $DB_PATH)
./scripts/dash phd        # Postgres leg (requires PHD_POSTGRES_URL after L-RF4)
./scripts/dash --url postgres://...    # any custom URL
make dash                 # alias for ./scripts/dash sqlite
make dash-phd             # alias for ./scripts/dash phd
```

If no `default = true` is set in config, rainfrog opens a picker on launch.

---

## Query

Sample IGLA queries to run from the editor pane (`Alt+2` to focus, `Alt+Enter`
to execute):

```sql
-- Champion + gate margin
SELECT format, algo, seed, hidden, lr, step, bpb, sha, gate_status
FROM bpb_samples
ORDER BY bpb ASC
LIMIT 1;

-- Format × algo grid
SELECT format, algo,
       MIN(bpb)        AS best_bpb,
       AVG(ema_bpb)    AS ema_bpb,
       COUNT(*)        AS samples
FROM bpb_samples
GROUP BY format, algo
ORDER BY best_bpb ASC;

-- Wave-8 queue progress
SELECT status, COUNT(*) FROM runs GROUP BY status;
```

For Postgres (after L-RF4):

```sql
SELECT count(*) FROM ssot.chapters;
SELECT name, length(body) FROM ssot.chapters ORDER BY name;
```

---

## Keybinds (cheat sheet)

### Pane focus

| Key          | Pane            |
| ------------ | --------------- |
| `Alt+1` / `Ctrl+k` | menu (schema browser) |
| `Alt+2` / `Ctrl+j` | query editor    |
| `Alt+3` / `Ctrl+h` | results         |
| `Alt+4` / `Ctrl+g` | query history   |
| `Alt+5` / `Ctrl+m` | favorites       |
| `Tab` / `Shift+Tab` | cycle focus    |
| `Ctrl+c`     | quit            |

### Query editor (vim-mode)

`i` insert · `Esc` normal · `Alt+Enter` / `F5` execute · `q` / `Alt+q` abort.
Movement: `h j k l`, `w e b`, `0 $`, `gg G`. Edit: `r y x p u Ctrl+r`.
`Ctrl+f` / `Alt+f` save current query to favorites.

### Schema browser (menu)

`j`/`k` move · `g`/`G` jump · `h`/`l` schemas/tables · `/` filter ·
`Enter` preview 100 rows · `R` reload.

### Results

`P` export CSV · `v` cell select · `V` row select · `y` copy ·
`g`/`G` top/bottom · `0`/`$` first/last column · `{`/`}` page up/down.

### History

`Alt+4` / `Ctrl+g` to focus · `y` copy · `I` edit in editor · `D` delete all.

---

## Keyring (password storage)

When connecting via `--host`/`--port`/`--username`/`--database` (no full URL),
rainfrog will prompt for a password and offer to save it to the OS keychain
(macOS Keychain / Linux Secret Service / Windows Credential Manager).
Decline if you prefer to inject the password via `DATABASE_URL`.

---

## Smoke test (G6 acceptance proof)

```bash
# Leg 1 — SQLite
make seed                                 # populates ./data.db
./scripts/dash sqlite
#  Alt+1 -> select bpb_samples -> Enter (preview 100 rows)
#  Alt+2 -> SELECT * FROM bpb_samples LIMIT 5;  -> Alt+Enter
#  Expect 2 champion rows (binary32/adamw/seed=43,44, bpb 2.1919 / 2.2024)

# Leg 2 — Postgres (after L-RF4)
PHD_POSTGRES_URL=postgres://phd:...@<railway-host>:<port>/ssot ./scripts/dash phd
#  Alt+2 -> SELECT count(*) FROM ssot.chapters; -> Alt+Enter
#  Expect a non-zero count from the PhD monograph SSOT.
```

Record an asciinema:

```bash
asciinema rec docs/runbooks/rainfrog-smoke.cast \
  --command "./scripts/dash sqlite" --title "rainfrog · IGLA SQLite leg"
```

---

## Troubleshoot

| Symptom                                          | Fix                                                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------- |
| `rainfrog: command not found`                    | `cargo install --locked rainfrog` (requires Rust 1.74+)                    |
| `Error: error returned from database: ... open`  | wrong SQLite path; check `DB_PATH` and that `npm run db:push` was run      |
| `Error: connection refused` on `phd-ssot`        | L-RF4 not done — Railway TCP proxy + volume mount missing                  |
| Mouse selection not working                      | `rainfrog -M true` (re-enables mouse; disables terminal copy/paste)        |
| Garbled UTF-8 in cells                           | terminal locale; export `LC_ALL=en_US.UTF-8`                               |
| Query history pane empty after restart           | history is per-session by design; star important queries with `Ctrl+f`     |
| Postgres `SCRAM-SHA-256` auth fails              | use full `connection_string` form, not split host/port — sqlx parses it    |

---

## Why rainfrog (not Drizzle Studio)

- R1 Rust-only — Drizzle Studio is JavaScript and pulls a hosted relay
  (`local.drizzle.studio`) into the loop.
- Two-DB native — single config covers SQLite + Postgres; rainfrog switches
  legs via picker. Drizzle Studio reads `drizzle.config.ts` and is
  schema-bound, not connection-bound.
- Vim ergonomics + query history + keyring — match the rest of the IGLA toolchain.
- Active upstream (last push 2026-04-24, 5,036★) — see ONE SHOT #99 sweep table.

`phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP`
