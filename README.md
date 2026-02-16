# agent-skills

Repositorio para centralizar skills reutilizables de agentes (Cursor/Codex) en un solo lugar.

## Objetivo

Este repo guarda el codigo fuente, documentacion y pruebas de cada skill para:

- versionar cambios con historial claro,
- facilitar mantenimiento y mejoras,
- reutilizar skills entre proyectos,
- evitar perdida de conocimiento operativo.

## Estructura del repositorio

Cada skill vive en su propia carpeta de primer nivel.

```text
.
├── README.md
├── .gitignore
└── podcast-maker/
    ├── SKILL.md
    ├── README.md
    ├── scripts/
    └── tests/
```

## Skills disponibles

### `podcast-maker`

Skill para generar guion y audio de podcast desde texto fuente, con:

- pipeline de generacion y post-procesado de script,
- sintetesis TTS con control por segmento,
- quality gates y validaciones,
- utilidades de checkpoint/debug y tests de regresion.

## Estandares para cada skill

Todo skill nuevo debe incluir, como minimo:

1. `SKILL.md` con la interfaz de uso del skill.
2. `README.md` con contexto tecnico y operacion local.
3. Carpeta `scripts/` con entrypoints y pipeline.
4. Pruebas en `tests/` (si aplica).

## Regla de higiene del repo

- No commitear artefactos temporales: checkpoints, logs, bundles, caches, `.DS_Store`, etc.
- Mantener `.gitignore` raiz y de cada skill alineados con sus artefactos reales.
- Evitar archivos locales de entorno (`.env`, credenciales, claves privadas).

## Como agregar un nuevo skill

1. Crear carpeta de primer nivel con nombre del skill (ej. `my-skill/`).
2. Agregar `SKILL.md`, `README.md`, `scripts/` y (opcional) `tests/`.
3. Definir o actualizar `.gitignore` para artefactos propios del skill.
4. Anadir una seccion en este `README.md` bajo "Skills disponibles".
5. Validar que `git status` quede limpio de temporales antes de commit.
