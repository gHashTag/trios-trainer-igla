# IGLA RACE — convenience targets.
# Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877

.PHONY: dash dash-phd seed push install-rainfrog

dash:
	./scripts/dash sqlite

dash-phd:
	./scripts/dash phd

seed:
	npm run db:push
	npm run db:seed

push:
	npm run db:push

install-rainfrog:
	cargo install --locked rainfrog
