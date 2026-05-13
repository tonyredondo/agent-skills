# setup-codex-prerequisites

Skill para preparar una estacion Windows con el baseline de herramientas que Codex suele necesitar para desarrollo, inspeccion de repositorios, paquetes, linting, auditoria ligera y builds.

## Que instala

- Bootstrap de `winget` cuando falta.
- Python 3.13 cuando no hay `python` ni `py`.
- Herramientas Python: `uv`, `uvx`, `pipx`, `PyYAML`.
- CLIs aisladas con `pipx`: `ruff`, `pytest`, `mypy`, `pre-commit`, `pip-audit`.
- Herramientas base: `git`, `gh`, `rg`, `pwsh`, `node`, `npm`, `pnpm`.
- Utilidades: `jq`, `yq`, `fd`, `fzf`, `bat`, `delta`, `7z`, `just`.
- Build y auditoria: `cmake`, `ninja`, `gitleaks`, `shellcheck`, `shfmt`, `hadolint`.

## Uso local

Desde la raiz de esta skill:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex-prerequisites.ps1
```

El script es idempotente: si una herramienta ya esta disponible y responde a su comprobacion de version, no se reinstala. Si un comando existe pero no pasa la comprobacion, intenta instalar o reparar el paquete correspondiente.

## Verificacion

El instalador termina verificando que cada comando esperado resuelve por nombre desde `PATH` y responde con version o salida de ayuda. Si una terminal estaba abierta antes de los cambios de `PATH`, puede ser necesario reiniciarla.

## Seguridad

Este skill no configura secretos, API keys ni integraciones especificas de servicios. Solo instala herramientas generales del entorno.
