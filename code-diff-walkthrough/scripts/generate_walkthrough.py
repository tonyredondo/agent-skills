#!/usr/bin/env python3
"""Generate bilingual annotated HTML walkthroughs for git diffs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EN_UI = {
    "how_to_read": "How to read this walkthrough",
    "purpose": "Purpose",
    "purpose_text": "Explain why each diff block exists: behavior, safety, integration points, tests, and validation.",
    "grounding": "Grounding",
    "grounding_text": "Generated from the real local diff, with optional plan/design context embedded for informed review.",
    "navigation": "Navigation",
    "navigation_text": "Use the left search to filter by file path or comment text. Each file can be collapsed independently.",
    "architecture": "Architecture",
    "architecture_text": "Use these flows to understand how the changed pieces call into each other before reviewing individual files.",
    "architecture_flows": "Call and data flows",
    "zoom_in": "Zoom in",
    "zoom_out": "Zoom out",
    "zoom_level": "Zoom level",
    "type_glossary": "Type glossary",
    "related_files": "related files",
    "focus_related_files": "Focus related files",
    "flow_focus": "Flow focus",
    "clear_flow_focus": "Clear flow focus",
    "when": "When",
    "evidence": "Evidence",
    "change_areas": "Change Areas",
    "commit_story": "Commit Story",
    "summary": "Summary",
    "files_tab": "Files",
    "file": "File",
    "changes": "Changes",
    "status": "Status",
    "file_rationale": "File rationale",
    "why_block": "Why this block changed",
    "changed_blocks": "Changed blocks detected",
    "filter": "Filter files or comments",
    "review_queue": "Review queue",
    "reviewed_progress_suffix": "reviewed",
    "expand_all": "Expand all",
    "collapse_all": "Collapse all",
    "expand": "Expand",
    "collapse": "Collapse",
    "files": "files",
    "context": "Context",
    "original_hunk": "Original hunk",
    "show_more_before": "Show earlier lines",
    "show_more_after": "Show later lines",
    "hide_extra": "Hide extra context",
    "new_lines": "new lines",
    "selection": "Selection",
    "selection_help": "Click any diff line to start a selection, then click another line in the same file to extend it.",
    "no_selection": "No line selected",
    "copy_reference": "Copy reference",
    "copy_review_prompt": "Copy review prompt",
    "clear_selection": "Clear",
    "add_comment_on": "Add a comment on",
    "edit_comment_on": "Edit comment on",
    "comment_placeholder": "Add your review comment",
    "cancel": "Cancel",
    "save": "Save",
    "edit": "Edit",
    "delete": "Delete",
    "copy_all_comments": "Copy all comments",
    "clear_all_comments": "Clear",
    "clear_all_comments_title": "Delete all saved comments",
    "no_comments_to_copy": "No comments to copy",
    "comments_copied": "Copied comments",
    "comments_cleared": "Cleared comments",
    "saved_comment": "Saved comment",
    "copied": "Copied",
    "review_prompt_prefix": "Please review this change and comment on any correctness, maintainability, or test risk:",
    "review_progress": "Review progress",
    "mark_reviewed": "Mark reviewed",
    "mark_pending": "Mark pending",
    "reviewed": "Reviewed",
    "pending": "Pending",
    "show_pending_only": "Show pending only",
    "clear_reviewed": "Clear reviewed",
    "hide_sidebar": "Hide navigation",
    "show_sidebar": "Show navigation",
}

ES_UI = {
    "how_to_read": "Cómo leer este walkthrough",
    "purpose": "Propósito",
    "purpose_text": "Explica por qué existe cada bloque del diff: comportamiento, seguridad, puntos de integración, tests y validación.",
    "grounding": "Base",
    "grounding_text": "Generado desde el diff local real, con contexto opcional de plan/diseño embebido para una revisión informada.",
    "navigation": "Navegación",
    "navigation_text": "Usa la búsqueda izquierda para filtrar por path de archivo o texto del comentario. Cada archivo se puede colapsar de forma independiente.",
    "architecture": "Arquitectura",
    "architecture_text": "Usa estos flujos para entender cómo se llaman entre sí las piezas modificadas antes de revisar archivos individuales.",
    "architecture_flows": "Flujos de llamadas y datos",
    "zoom_in": "Acercar",
    "zoom_out": "Alejar",
    "zoom_level": "Nivel de zoom",
    "type_glossary": "Glosario de tipos",
    "related_files": "archivos relacionados",
    "focus_related_files": "Filtrar archivos relacionados",
    "flow_focus": "Filtro de flujo",
    "clear_flow_focus": "Limpiar filtro de flujo",
    "when": "Cuando",
    "evidence": "Evidencia",
    "change_areas": "Áreas de cambio",
    "commit_story": "Historia de commits",
    "summary": "Resumen",
    "files_tab": "Archivos",
    "file": "Archivo",
    "changes": "Cambios",
    "status": "Estado",
    "file_rationale": "Razonamiento del archivo",
    "why_block": "Por qué cambió este bloque",
    "changed_blocks": "Bloques detectados",
    "filter": "Filtrar archivos o comentarios",
    "review_queue": "Cola de review",
    "reviewed_progress_suffix": "revisados",
    "expand_all": "Expandir todo",
    "collapse_all": "Colapsar todo",
    "expand": "Expandir",
    "collapse": "Colapsar",
    "files": "archivos",
    "context": "Contexto",
    "original_hunk": "Hunk original",
    "show_more_before": "Mostrar líneas anteriores",
    "show_more_after": "Mostrar líneas posteriores",
    "hide_extra": "Ocultar contexto extra",
    "new_lines": "líneas nuevas",
    "selection": "Selección",
    "selection_help": "Haz click en una línea del diff para iniciar una selección y luego en otra línea del mismo archivo para extenderla.",
    "no_selection": "Ninguna línea seleccionada",
    "copy_reference": "Copiar referencia",
    "copy_review_prompt": "Copiar prompt de review",
    "clear_selection": "Limpiar",
    "add_comment_on": "Añadir comentario en",
    "edit_comment_on": "Editar comentario en",
    "comment_placeholder": "Añade tu comentario de review",
    "cancel": "Cancelar",
    "save": "Guardar",
    "edit": "Editar",
    "delete": "Eliminar",
    "copy_all_comments": "Copiar todos los comentarios",
    "clear_all_comments": "Borrar",
    "clear_all_comments_title": "Borrar todos los comentarios guardados",
    "no_comments_to_copy": "No hay comentarios para copiar",
    "comments_copied": "Comentarios copiados",
    "comments_cleared": "Comentarios borrados",
    "saved_comment": "Comentario guardado",
    "copied": "Copiado",
    "review_prompt_prefix": "Revisa este cambio y comenta cualquier riesgo de correctitud, mantenibilidad o tests:",
    "review_progress": "Progreso de review",
    "mark_reviewed": "Marcar revisado",
    "mark_pending": "Marcar pendiente",
    "reviewed": "Revisado",
    "pending": "Pendiente",
    "show_pending_only": "Mostrar solo pendientes",
    "clear_reviewed": "Limpiar revisados",
    "hide_sidebar": "Ocultar navegación",
    "show_sidebar": "Mostrar navegación",
}


CATEGORIES = [
    (
        re.compile(r"(^|/)test(s)?/|Test|Tests|Spec|Fixture|\.Tests?\.", re.I),
        "Tests",
        "Tests",
        "These changes verify the new behavior and protect the edge cases introduced by the implementation.",
        "Estos cambios verifican el nuevo comportamiento y protegen los casos borde introducidos por la implementación.",
    ),
    (
        re.compile(r"doc|readme|\.md$", re.I),
        "Documentation",
        "Documentación",
        "This documents the behavior, constraints, or reviewer-facing context for the change.",
        "Esto documenta el comportamiento, las restricciones o el contexto útil para revisar el cambio.",
    ),
    (
        re.compile(r"config|settings|options|yaml|json|toml|props|targets", re.I),
        "Configuration",
        "Configuración",
        "This changes configuration or build wiring so the feature is enabled with the right scope.",
        "Esto cambia configuración o wiring de build para activar la feature con el scope correcto.",
    ),
    (
        re.compile(r"client|api|request|response|payload|http|net|backend", re.I),
        "API/backend contract",
        "Contrato API/backend",
        "This wires the feature into an external or backend contract and keeps request/response semantics explicit.",
        "Esto conecta la feature con un contrato externo o de backend y mantiene explícita la semántica de request/response.",
    ),
    (
        re.compile(r"ipc|message|channel|process|runner|host|worker", re.I),
        "Process/runtime wiring",
        "Wiring de proceso/runtime",
        "This moves data across runtime or process boundaries so the final behavior is based on the complete execution.",
        "Esto mueve datos a través de límites de runtime o proceso para que el comportamiento final use la ejecución completa.",
    ),
    (
        re.compile(r"coverage|report|xml|cobertura|opencover|jacoco|coverlet", re.I),
        "Coverage/reporting",
        "Coverage/reporte",
        "This changes coverage/reporting behavior, usually by preserving line-level data instead of relying only on a final percentage.",
        "Esto cambia comportamiento de coverage/reporte, normalmente conservando datos a nivel de línea en vez de depender solo de un porcentaje final.",
    ),
    (
        re.compile(r"cache|store|persist|state|database|repository", re.I),
        "State/cache",
        "Estado/cache",
        "This controls how state is stored, reused, or invalidated so later steps do not consume stale or unsafe data.",
        "Esto controla cómo se guarda, reutiliza o invalida el estado para que pasos posteriores no consuman datos obsoletos o inseguros.",
    ),
]


PATTERNS = [
    (
        re.compile(r"\b(?:TODO|FIXME|throw|Exception|Error|Warning|fail|invalid|malformed|missing)\b", re.I),
        "This block changes an error or failure path so the flow handles that condition explicitly.",
        "Este bloque cambia una ruta de error o fallo para que el flujo trate esa condición explícitamente.",
    ),
    (
        re.compile(r"\b(?:cache|cached|key|scope|salt|hash|persist|store|load|save)\b", re.I),
        "This block changes how scoped or persisted state is stored, loaded, or reused.",
        "Este bloque cambia cómo se guarda, carga o reutiliza estado persistido o con scope.",
    ),
    (
        re.compile(r"\b(?:request|response|payload|metadata|meta|json|serialize|deserialize)\b", re.I),
        "This block changes request, response, or serialized data that another part of the flow consumes.",
        "Este bloque cambia datos de request, respuesta o serialización que consume otra parte del flujo.",
    ),
    (
        re.compile(r"\b(?:skip|skippable|filter|candidate|disabled|quarantine)\b", re.I),
        "This block changes skip decision or filtering data used to decide which tests continue through the flow.",
        "Este bloque cambia datos de decisión de skip o filtrado usados para decidir qué tests continúan en el flujo.",
    ),
    (
        re.compile(r"\b(?:merge|aggregate|union|combine|priority|selected)\b|\|\||\?\?", re.I),
        "This block combines or prioritizes multiple values before the rest of the flow uses the result.",
        "Este bloque combina o prioriza varios valores antes de que el resto del flujo use el resultado.",
    ),
    (
        re.compile(r"\b(?:line|bitmap|coverage|covered|executable|hit|count|percentage|percent)\b", re.I),
        "This block changes coverage or line-level data used by the reporting calculation.",
        "Este bloque cambia datos de coverage o de líneas usados por el cálculo del reporte.",
    ),
    (
        re.compile(r"\b(?:xml|cobertura|opencover|jacoco|coverlet|vanguard|report)\b", re.I),
        "This block changes handling for an external report or tool-specific coverage format.",
        "Este bloque cambia el manejo de un reporte externo o de un formato de coverage de una herramienta.",
    ),
    (
        re.compile(r"\b(?:test|assert|should|expected|fixture|smoke)\b", re.I),
        "This test or assertion covers the changed behavior or a regression case.",
        "Este test o aserción cubre el comportamiento cambiado o un caso de regresión.",
    ),
    (
        re.compile(r"\b(?:env|environment|variable|config|setting|option)\b", re.I),
        "This passes configuration through the process environment so child processes can read it.",
        "Esto pasa configuración por el entorno del proceso para que los procesos hijos puedan leerla.",
    ),
]


def run(repo: Path, args: list[str], check: bool = True) -> str:
    proc = subprocess.run(args, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout}")
    return proc.stdout


def parse_pr_ref(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"/pull/(\d+)", value)
    return match.group(1) if match else value


def resolve_refs(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    pr = None
    if args.pr:
        pr_ref = parse_pr_ref(args.pr)
        pr = json.loads(run(repo, ["gh", "pr", "view", pr_ref, "--json", "title,number,url,headRefName,baseRefName,headRefOid,isDraft"]))
        base = args.base or pr["baseRefName"]
        head = args.head or pr["headRefName"]
        diff_range = args.range or f"{base}...{head}"
    else:
        base = args.base
        head = args.head
        diff_range = args.range or f"{base}...{head}"

    head_sha = run(repo, ["git", "rev-parse", head]).strip() if head else run(repo, ["git", "rev-parse", "HEAD"]).strip()
    merge_base = ""
    if "..." in diff_range:
        left, right = diff_range.split("...", 1)
        merge_base = run(repo, ["git", "merge-base", left, right]).strip()

    return {
        "pr": pr,
        "base": base,
        "head": head,
        "range": diff_range,
        "head_sha": head_sha,
        "short_head": run(repo, ["git", "rev-parse", "--short=10", head_sha]).strip(),
        "merge_base": merge_base,
    }


def parse_numstat(text: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            result[parts[2]] = {"additions": parts[0], "deletions": parts[1]}
    return result


def parse_diff(text: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    hunk: dict[str, Any] | None = None
    old_no: int | None = None
    new_no: int | None = None

    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            if cur:
                if hunk:
                    cur["hunks"].append(hunk)
                files.append(cur)
            parts = raw.split(" ")
            old_path = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else parts[2] if len(parts) > 2 else ""
            new_path = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else parts[3] if len(parts) > 3 else old_path
            cur = {"old_path": old_path, "new_path": new_path, "headers": [raw], "hunks": [], "status": "modified"}
            hunk = None
            continue
        if cur is None:
            continue
        if raw.startswith("new file mode"):
            cur["status"] = "added"
            cur["headers"].append(raw)
            continue
        if raw.startswith("deleted file mode"):
            cur["status"] = "deleted"
            cur["headers"].append(raw)
            continue
        if raw.startswith(("rename from", "rename to", "index ", "--- ", "+++ ")):
            cur["headers"].append(raw)
            continue
        if raw.startswith("@@"):
            if hunk:
                cur["hunks"].append(hunk)
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", raw)
            old_no = int(match.group(1)) if match else None
            new_no = int(match.group(2)) if match else None
            hunk = {"header": raw, "lines": []}
            continue
        if hunk is None:
            cur["headers"].append(raw)
            continue
        kind = raw[:1]
        content = raw[1:] if raw else ""
        if kind == "+":
            hunk["lines"].append({"kind": "add", "old": None, "new": new_no, "text": content})
            new_no = new_no + 1 if new_no is not None else None
        elif kind == "-":
            hunk["lines"].append({"kind": "del", "old": old_no, "new": None, "text": content})
            old_no = old_no + 1 if old_no is not None else None
        elif kind == " ":
            hunk["lines"].append({"kind": "ctx", "old": old_no, "new": new_no, "text": content})
            old_no = old_no + 1 if old_no is not None else None
            new_no = new_no + 1 if new_no is not None else None
        else:
            hunk["lines"].append({"kind": "meta", "old": None, "new": None, "text": raw})
    if cur:
        if hunk:
            cur["hunks"].append(hunk)
        files.append(cur)
    return files


def read_file_at_ref(repo: Path, ref: str, path: str) -> list[str]:
    if not path or path == "/dev/null":
        return []
    try:
        return run(repo, ["git", "show", f"{ref}:{path}"]).splitlines()
    except RuntimeError:
        local_path = repo / path
        if local_path.exists() and local_path.is_file():
            return local_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return []


def load_context(paths: list[str]) -> str:
    chunks = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks.append(f"# {path}\n{text[:30000]}")
    return "\n\n".join(chunks)


def load_notes(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def localized(value: Any, lang: str, fallback: str = "") -> str:
    """Returns a localized string from a plain value or an {en, es} mapping."""
    if isinstance(value, dict):
        for key in (lang, "en", "es"):
            item = value.get(key)
            if item is not None:
                return str(item)
        return fallback
    if value is None:
        return fallback
    return str(value)


def localized_pair(value: Any, fallback_en: str = "", fallback_es: str = "") -> dict[str, str]:
    """Normalizes a plain or localized value into the bilingual shape used by the HTML payload."""
    return {
        "en": localized(value, "en", fallback_en),
        "es": localized(value, "es", fallback_es or fallback_en),
    }


def normalize_id(value: Any, fallback: str) -> str:
    """Builds a stable HTML/data id for architecture nodes, sections, and edges."""
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = text.strip("-")
    return text or fallback


def normalize_files(value: Any) -> list[str]:
    """Normalizes a notes-json file reference into a list of repository paths."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def normalize_int(value: Any, fallback: int) -> int:
    """Reads a positive integer layout value, falling back when the value is absent or invalid."""
    try:
        parsed = int(value)
        return parsed if parsed > 0 else fallback
    except (TypeError, ValueError):
        return fallback


def normalize_evidence(value: Any) -> list[dict[str, Any]]:
    """Normalizes source evidence references attached to architecture nodes or glossary entries."""
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    evidence = []
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            file_path = item
            line = None
            match = re.match(r"^(.*):(\d+)$", item)
            if match:
                file_path = match.group(1)
                line = int(match.group(2))
            evidence.append(
                {
                    "label": localized_pair(item),
                    "file": file_path,
                    "line": line,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        file_path = str(item.get("file") or item.get("path") or "")
        line = item.get("line") or item.get("start")
        try:
            line = int(line) if line is not None else None
        except (TypeError, ValueError):
            line = None
        label = item.get("label") or item.get("description") or (f"{file_path}:{line}" if line else file_path) or f"evidence-{index + 1}"
        evidence.append(
            {
                "label": localized_pair(label),
                "file": file_path,
                "line": line,
            }
        )
    return evidence


def normalize_architecture(notes_data: dict[str, Any]) -> dict[str, Any]:
    """Normalizes optional notes-json architecture flows into a render-safe, bilingual payload."""
    raw = notes_data.get("architecture")
    if not raw:
        return {"sections": [], "glossary": []}

    if isinstance(raw, list):
        raw = {"sections": raw}
    if not isinstance(raw, dict):
        return {"sections": [], "glossary": []}

    sections = []
    raw_sections = raw.get("sections") or raw.get("flows") or []
    if isinstance(raw_sections, dict):
        raw_sections = [raw_sections]

    for section_index, section_raw in enumerate(raw_sections):
        if not isinstance(section_raw, dict):
            continue

        nodes = []
        raw_nodes = section_raw.get("nodes") or []
        if not isinstance(raw_nodes, list):
            raw_nodes = []

        for node_index, node_raw in enumerate(raw_nodes):
            if not isinstance(node_raw, dict):
                continue
            node_id = normalize_id(node_raw.get("id") or node_raw.get("label"), f"node-{node_index + 1}")
            node = {
                "id": node_id,
                "label": localized_pair(node_raw.get("label") or node_id, node_id, node_id),
                "detail": localized_pair(node_raw.get("detail") or node_raw.get("description")),
                "when": localized_pair(node_raw.get("when") or node_raw.get("condition")),
                "evidence": normalize_evidence(node_raw.get("evidence")),
                "kind": normalize_id(node_raw.get("kind"), "process"),
                "files": normalize_files(node_raw.get("files") or node_raw.get("file")),
                "row": normalize_int(node_raw.get("row"), 1),
                "column": normalize_int(node_raw.get("column") or node_raw.get("col"), node_index + 1),
            }
            nodes.append(node)

        node_ids = {node["id"] for node in nodes}
        edges = []
        raw_edges = section_raw.get("edges")
        if raw_edges is None and len(nodes) > 1:
            raw_edges = [
                {"from": nodes[i]["id"], "to": nodes[i + 1]["id"]}
                for i in range(len(nodes) - 1)
            ]
        if isinstance(raw_edges, list):
            for edge_raw in raw_edges:
                if not isinstance(edge_raw, dict):
                    continue
                source = normalize_id(edge_raw.get("from") or edge_raw.get("source"), "")
                target = normalize_id(edge_raw.get("to") or edge_raw.get("target"), "")
                if not source or not target or source not in node_ids or target not in node_ids:
                    continue
                edges.append(
                    {
                        "from": source,
                        "to": target,
                        "label": localized_pair(edge_raw.get("label")),
                        "when": localized_pair(edge_raw.get("when") or edge_raw.get("condition")),
                        "evidence": normalize_evidence(edge_raw.get("evidence")),
                    }
                )

        if not nodes:
            continue

        columns = max(max(node["column"] for node in nodes), 1)
        sections.append(
            {
                "id": normalize_id(section_raw.get("id") or section_raw.get("title"), f"flow-{section_index + 1}"),
                "title": localized_pair(section_raw.get("title"), f"Flow {section_index + 1}", f"Flujo {section_index + 1}"),
                "summary": localized_pair(section_raw.get("summary") or section_raw.get("description")),
                "columns": columns,
                "nodes": nodes,
                "edges": edges,
            }
        )

    glossary = []
    raw_glossary = raw.get("glossary") or raw.get("types") or []
    if isinstance(raw_glossary, dict):
        raw_glossary = [
            {"term": key, "description": value}
            for key, value in raw_glossary.items()
        ]
    if isinstance(raw_glossary, list):
        for item_index, item in enumerate(raw_glossary):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term") or item.get("name") or f"Type {item_index + 1}")
            glossary.append(
                {
                    "term": term,
                    "description": localized_pair(item.get("description") or item.get("detail")),
                    "when": localized_pair(item.get("when") or item.get("condition")),
                    "evidence": normalize_evidence(item.get("evidence")),
                    "files": normalize_files(item.get("files") or item.get("file")),
                }
            )

    return {
        "title": localized_pair(raw.get("title"), "Architecture", "Arquitectura"),
        "summary": localized_pair(raw.get("summary") or raw.get("description")),
        "sections": sections,
        "glossary": glossary,
    }


def categorize(path: str) -> tuple[str, str, str, str]:
    for regex, en, es, note_en, note_es in CATEGORIES:
        if regex.search(path):
            return en, es, note_en, note_es
    return (
        "Implementation",
        "Implementación",
        "This file participates in the main implementation wiring for the change.",
        "Este archivo participa en el wiring principal de implementación del cambio.",
    )


def symbols(lines: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    patterns = [
        r"\b(class|record|struct|enum|interface)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b(public|private|internal|protected)\s+(?:static\s+)?(?:async\s+)?(?:[A-Za-z0-9_<>,\[\]?]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\b(function|def|func)\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]
    for line in lines:
        if line["kind"] != "add":
            continue
        text = line["text"].strip()
        if text.startswith(("//", "///", "#", "*")):
            continue
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(2)
                if name not in found:
                    found.append(name)
                break
        if len(found) >= 8:
            break
    return found


def detect_symbol_context(file_lines: list[str], line_no: int | None) -> str:
    if not file_lines or not line_no:
        return ""

    class_stack: list[str] = []
    method_name = ""
    class_pattern = re.compile(r"\b(?:class|record|struct|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
    method_pattern = re.compile(
        r"^\s*(?:(?:public|private|internal|protected)\s+)?"
        r"(?:(?:static|sealed|virtual|override|async|extern|partial)\s+)*"
        r"(?:[A-Za-z_][A-Za-z0-9_<>,\[\]\.?]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    python_pattern = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
    js_pattern = re.compile(r"\b(?:function\s+([A-Za-z_][A-Za-z0-9_]*)|(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\()")

    for raw in file_lines[: min(len(file_lines), line_no + 1)]:
        text = raw.strip()
        if not text or text.startswith(("//", "///", "#", "*")):
            continue
        if text.startswith(("return ", "var ", "new ", "if ", "for ", "foreach ", "while ", "switch ", "catch ", "using ", "lock ")):
            continue

        class_match = class_pattern.search(text)
        if class_match:
            name = class_match.group(1)
            if name not in class_stack:
                class_stack = [name]

        method_match = method_pattern.search(text)
        if method_match:
            candidate = method_match.group(1)
            if candidate not in {"if", "for", "foreach", "while", "switch", "catch", "using", "lock"}:
                method_name = candidate
                continue

        python_match = python_pattern.search(raw)
        if python_match:
            name = python_match.group(1)
            if text.startswith("class "):
                class_stack = [name]
            else:
                method_name = name
            continue

        js_match = js_pattern.search(text)
        if js_match:
            method_name = js_match.group(1) or js_match.group(2) or method_name

    if class_stack and method_name:
        return ".".join(class_stack + [method_name])
    if method_name:
        return method_name
    if class_stack:
        return ".".join(class_stack)
    return ""


def line_range_for_hunk(hunk: dict[str, Any]) -> tuple[int | None, int | None]:
    new_lines = [line["new"] for line in hunk["lines"] if line.get("new") is not None]
    if not new_lines:
        return None, None
    return min(new_lines), max(new_lines)


def extra_context_lines(file_lines: list[str], start: int | None, end: int | None, radius: int) -> dict[str, list[dict[str, Any]]]:
    if not file_lines or not start or not end or radius <= 0:
        return {"before": [], "after": []}

    before_start = max(1, start - radius)
    before = [
        {"kind": "extra", "old": None, "new": line_no, "text": file_lines[line_no - 1]}
        for line_no in range(before_start, start)
    ]

    after_end = min(len(file_lines), end + radius)
    after = [
        {"kind": "extra", "old": None, "new": line_no, "text": file_lines[line_no - 1]}
        for line_no in range(end + 1, after_end + 1)
    ]

    return {"before": before, "after": after}


def stable_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_state_key(repo: Path, refs: dict[str, Any]) -> str:
    if refs.get("pr"):
        scope = {"repo": str(repo), "pr": refs["pr"].get("url") or refs["pr"].get("number")}
    else:
        scope = {"repo": str(repo), "range": refs["range"], "base": refs["base"], "head": refs["head"]}
    return stable_digest(scope)[:24]


def review_fingerprint(file: dict[str, Any]) -> str:
    payload = {
        "old_path": file.get("old_path"),
        "new_path": file.get("new_path"),
        "status": file.get("status"),
        "headers": file.get("headers", []),
        "hunks": [
            {
                "header": hunk.get("header"),
                "lines": hunk.get("lines", []),
            }
            for hunk in file.get("hunks", [])
        ],
    }
    return stable_digest(payload)


def note_from_custom(path: str, body: str, notes_data: dict[str, Any], lang: str) -> list[str]:
    """Return hunk-level notes from external pattern mappings."""
    notes: list[str] = []
    for item in notes_data.get("patterns", []):
        regex = item.get("regex")
        if regex and re.search(regex, body, re.I | re.M):
            notes.append(item.get(lang, item.get("note", "")))
    return [n for n in notes if n]


def changed_lines(hunk: dict[str, Any]) -> list[str]:
    """Return only the added and removed source lines that define the actual diff."""
    return [line["text"] for line in hunk["lines"] if line["kind"] in {"add", "del"}]


def format_series(values: list[str], conjunction: str) -> str:
    """Format a short human-readable series without adding punctuation-heavy noise."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} {conjunction} {values[-1]}"


def extract_configuration_keys(body: str) -> list[str]:
    """Extract ConfigurationKeys references from changed lines for concrete review notes."""
    keys: list[str] = []
    for match in re.finditer(r"ConfigurationKeys\.([A-Za-z0-9_.]+)\]", body):
        value = match.group(1)
        if value and value not in keys:
            keys.append(value)
    return keys


def environment_assignment_notes(body: str) -> dict[str, list[str]]:
    """Describe simple environment-variable assignments using the names found in the diff."""
    if not re.search(r"EnvironmentVariables\s*\[|environmentVariables\s*\[", body):
        return {"en": [], "es": []}

    keys = [f"`{key}`" for key in extract_configuration_keys(body)]
    if keys:
        en_keys = format_series(keys, "and")
        es_keys = format_series(keys, "y")
        return {
            "en": [f"This adds environment entries for {en_keys} so spawned processes receive those values."],
            "es": [f"Esto añade entradas de entorno para {es_keys}, de forma que los procesos lanzados reciban esos valores."],
        }

    return {
        "en": ["This adds environment entries so spawned processes receive the values introduced by this change."],
        "es": ["Esto añade entradas de entorno para que los procesos lanzados reciban los valores introducidos por este cambio."],
    }


def is_import_statement_line(line: str) -> bool:
    """Return true for simple import/using declaration lines."""
    value = line.strip()
    if not value:
        return True
    return bool(
        re.match(r"^using\s+(?:static\s+)?[A-Za-z_][A-Za-z0-9_.]*(?:\s*=\s*[A-Za-z_][A-Za-z0-9_.]*)?\s*;\s*$", value)
        or re.match(r"^import\s+(?:static\s+)?[A-Za-z_][A-Za-z0-9_.]*(?:\.\*)?\s*;\s*$", value)
        or re.match(r"^from\s+[A-Za-z_][A-Za-z0-9_.]*\s+import\s+.+$", value)
        or re.match(r"^import\s+.+\s+from\s+['\"].+['\"]\s*;?\s*$", value)
    )


def format_import_statement(line: str) -> str:
    """Return a compact import target for review notes."""
    value = line.strip().rstrip(";")
    value = re.sub(r"^using\s+", "", value)
    value = re.sub(r"^import\s+static\s+", "", value)
    value = re.sub(r"^import\s+", "", value)
    return f"`{value}`" if value else "`import`"


def import_change_notes(hunk: dict[str, Any]) -> dict[str, list[str]]:
    """Describe import-only hunks without reusing broad file rationale."""
    changed = changed_lines(hunk)
    non_empty = [line for line in changed if line.strip()]
    if not non_empty or not all(is_import_statement_line(line) for line in changed):
        return {"en": [], "es": []}

    added = [format_import_statement(line) for line in changed_texts_by_kind(hunk, "add") if line.strip()]
    removed = [format_import_statement(line) for line in changed_texts_by_kind(hunk, "del") if line.strip()]
    if added and not removed:
        en_imports = format_series(added, "and")
        es_imports = format_series(added, "y")
        return {
            "en": [f"This hunk has no runtime behavior by itself; it only brings {en_imports} into scope."],
            "es": [f"Este hunk no tiene comportamiento runtime por si solo; solo pone {es_imports} en scope."],
        }

    if removed and not added:
        en_imports = format_series(removed, "and")
        es_imports = format_series(removed, "y")
        return {
            "en": [f"This removes the now-unused {en_imports} import; the behavior change is in the code hunks that no longer need it."],
            "es": [f"Esto elimina el import {es_imports} que ya no se usa; el cambio de comportamiento esta en los hunks de codigo que dejaron de necesitarlo."],
        }

    return {
        "en": ["This updates import declarations so the file compiles with the changed dependencies; behavior is explained in the later code hunks."],
        "es": ["Esto actualiza declaraciones de imports para que el fichero compile con las dependencias cambiadas; el comportamiento se explica en los hunks posteriores."],
    }


def changed_texts_by_kind(hunk: dict[str, Any], kind: str) -> list[str]:
    """Return changed source lines for a specific diff kind."""
    return [line["text"] for line in hunk["lines"] if line["kind"] == kind]


def extract_if_condition(line: str) -> str:
    """Extract the condition from a simple single-line if statement."""
    match = re.match(r"^\s*if\s*\((.*)\)\s*$", line)
    return match.group(1).strip() if match else ""


def added_condition_terms(old_condition: str, new_condition: str) -> list[str]:
    """Find simple OR terms that were added to an existing condition."""
    old_terms = {part.strip() for part in old_condition.split("||") if part.strip()}
    new_terms = [part.strip() for part in new_condition.split("||") if part.strip()]
    return [term for term in new_terms if term not in old_terms]


def condition_change_notes(hunk: dict[str, Any], body: str) -> dict[str, list[str]]:
    """Describe simple guard changes without falling back to broad keyword notes."""
    removed_conditions = [condition for condition in (extract_if_condition(line) for line in changed_texts_by_kind(hunk, "del")) if condition]
    added_conditions = [condition for condition in (extract_if_condition(line) for line in changed_texts_by_kind(hunk, "add")) if condition]
    if len(removed_conditions) != 1 or len(added_conditions) != 1:
        return {"en": [], "es": []}

    old_condition = removed_conditions[0]
    new_condition = added_conditions[0]
    if old_condition == new_condition:
        return {"en": [], "es": []}

    added_terms = added_condition_terms(old_condition, new_condition)
    if added_terms:
        en_terms = format_series([f"`{term}`" for term in added_terms], "or")
        es_terms = format_series([f"`{term}`" for term in added_terms], "o")
        if re.search(r"\bcoverage\b", body, re.I):
            return {
                "en": [f"This widens the coverage-output guard: the block still runs for `{old_condition}`, and now also runs for {en_terms}."],
                "es": [f"Esto amplía el guard del output de coverage: el bloque sigue ejecutándose con `{old_condition}` y ahora también con {es_terms}."],
            }
        return {
            "en": [f"This widens the guard from `{old_condition}` to `{new_condition}`, adding {en_terms} as an allowed path."],
            "es": [f"Esto amplía el guard de `{old_condition}` a `{new_condition}`, añadiendo {es_terms} como ruta permitida."],
        }

    return {
        "en": [f"This changes the guard from `{old_condition}` to `{new_condition}`."],
        "es": [f"Esto cambia el guard de `{old_condition}` a `{new_condition}`."],
    }


def hunk_notes(path: str, hunk: dict[str, Any], file_note_en: str, file_note_es: str, notes_data: dict[str, Any]) -> dict[str, list[str]]:
    changed = changed_lines(hunk)
    body = "\n".join(changed)

    specific_notes = import_change_notes(hunk)
    if not specific_notes["en"]:
        specific_notes = condition_change_notes(hunk, body)
    if not specific_notes["en"]:
        specific_notes = environment_assignment_notes(body)

    if specific_notes["en"]:
        notes = specific_notes
    else:
        notes = {"en": note_from_custom(path, body, notes_data, "en"), "es": note_from_custom(path, body, notes_data, "es")}

    if specific_notes["en"]:
        max_notes = 1
    elif notes["en"]:
        max_notes = 1
    elif len(changed) <= 3:
        max_notes = 2
    else:
        max_notes = 3

    for regex, en, es in PATTERNS:
        if specific_notes["en"] or notes["en"]:
            break
        if regex.search(body):
            notes["en"].append(en)
            notes["es"].append(es)
    if not notes["en"]:
        notes["en"].append(file_note_en)
    if not notes["es"]:
        notes["es"].append(file_note_es)
    for lang in ("en", "es"):
        dedup: list[str] = []
        for note in notes[lang]:
            if note and note not in dedup:
                dedup.append(note)
        notes[lang] = dedup[:max_notes]
    return notes


def build_model(repo: Path, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    refs = resolve_refs(repo, args)
    diff_text = run(repo, ["git", "diff", "--find-renames", "--src-prefix=a/", "--dst-prefix=b/", refs["range"]])
    stat_text = run(repo, ["git", "diff", "--stat", refs["range"]])
    numstat_text = run(repo, ["git", "diff", "--numstat", refs["range"]])
    commits = run(repo, ["git", "log", "--oneline", "--no-merges", refs["range"].replace("...", "..")], check=False).strip().splitlines()
    numstat = parse_numstat(numstat_text)
    notes_data = load_notes(args.notes_json)
    context = load_context(args.context or [])
    files = parse_diff(diff_text)

    for file in files:
        path = file["new_path"]
        file_lines = read_file_at_ref(repo, refs["head_sha"], path)
        cat_en, cat_es, note_en, note_es = categorize(path)
        file_notes = notes_data.get("files", {})
        if path in file_notes:
            custom = file_notes[path]
            if isinstance(custom, dict):
                note_en = custom.get("en", note_en)
                note_es = custom.get("es", note_es)
                category = custom.get("category")
                if isinstance(category, dict):
                    cat_en = str(category.get("en", cat_en))
                    cat_es = str(category.get("es", cat_es))
                elif isinstance(category, str):
                    cat_en = category
                    cat_es = category
            elif isinstance(custom, str):
                note_en = custom
                note_es = custom
        file["category"] = {"en": cat_en, "es": cat_es}
        file["note"] = {"en": note_en, "es": note_es}
        file["numstat"] = numstat.get(path, numstat.get(file["old_path"], {"additions": "?", "deletions": "?"}))
        file["reviewFingerprint"] = review_fingerprint(file)
        for hunk in file["hunks"]:
            start_line, end_line = line_range_for_hunk(hunk)
            symbol_context = detect_symbol_context(file_lines, start_line)
            context_lines = extra_context_lines(file_lines, start_line, end_line, args.surrounding_lines)
            hunk["comment"] = {
                "notes": hunk_notes(path, hunk, note_en, note_es, notes_data),
                "symbols": symbols(hunk["lines"]),
                "adds": sum(1 for line in hunk["lines"] if line["kind"] == "add"),
                "dels": sum(1 for line in hunk["lines"] if line["kind"] == "del"),
            }
            hunk["context"] = {
                "symbol": symbol_context,
                "newStart": start_line,
                "newEnd": end_line,
                "before": context_lines["before"],
                "after": context_lines["after"],
            }

    categories: dict[str, dict[str, dict[str, int]]] = {"en": {}, "es": {}}
    for file in files:
        for lang in ("en", "es"):
            name = file["category"][lang]
            categories[lang].setdefault(name, {"files": 0, "add": 0, "del": 0})
            categories[lang][name]["files"] += 1
            try:
                categories[lang][name]["add"] += int(file["numstat"]["additions"])
                categories[lang][name]["del"] += int(file["numstat"]["deletions"])
            except ValueError:
                pass

    outdir = Path(args.out).expanduser() if args.out else Path(tempfile.mkdtemp(prefix="code-diff-walkthrough-"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pr.diff").write_text(diff_text, encoding="utf-8")
    (outdir / "diff-stat.txt").write_text(stat_text, encoding="utf-8")
    (outdir / "commits.txt").write_text("\n".join(commits) + ("\n" if commits else ""), encoding="utf-8")
    if context:
        (outdir / "context.txt").write_text(context, encoding="utf-8")

    title = args.title or (refs["pr"]["title"] if refs["pr"] else f"Diff {refs['range']}")
    model = {
        "title": title,
        "repo": str(repo),
        "base": refs["base"],
        "head": refs["head"],
        "range": refs["range"],
        "headSha": refs["head_sha"],
        "shortHead": refs["short_head"],
        "mergeBase": refs["merge_base"],
        "reviewStateKey": review_state_key(repo, refs),
        "pr": refs["pr"],
        "generatedAt": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "commitLog": commits,
        "statText": stat_text,
        "contextSummary": context[:2000],
        "categories": categories,
        "architecture": normalize_architecture(notes_data),
        "files": files,
    }
    return model, outdir


def render_html(model: dict[str, Any], lang: str) -> str:
    ui = EN_UI if lang == "en" else ES_UI
    payload = dict(model)
    payload["language"] = lang
    payload["ui"] = ui
    json_payload = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page_title = f"PR #{model['pr']['number']} Walkthrough" if model.get("pr") else "Diff Walkthrough"
    if lang == "es":
        page_title = f"Walkthrough PR #{model['pr']['number']}" if model.get("pr") else "Walkthrough del Diff"
    highlight_base = "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build"
    highlight_languages = [
        "csharp",
        "go",
        "java",
        "typescript",
        "fsharp",
        "powershell",
        "vbnet",
        "kotlin",
        "swift",
        "scala",
        "dockerfile",
        "makefile",
        "ini",
    ]
    highlight_assets = "\n".join(
        [
            f'<link rel="stylesheet" href="{highlight_base}/styles/github.min.css">',
            f'<script defer src="{highlight_base}/highlight.min.js"></script>',
            *(f'<script defer src="{highlight_base}/languages/{language}.min.js"></script>' for language in highlight_languages),
        ]
    )

    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page_title)} - {html.escape(model['title'])}</title>
{highlight_assets}
<style>
:root {{ color-scheme: light; --bg:#f5f7fb; --surface:#ffffff; --surface2:#f9fbff; --surface3:#eef3f9; --border:#d8e0ea; --border-strong:#c7d1dc; --text:#172033; --muted:#64748b; --muted2:#94a3b8; --blue:#2563eb; --blue-soft:#e8f0ff; --title:#1d4ed8; --violet:#6d5bd0; --green:#e9f8ef; --green2:#15803d; --red:#fff1f2; --red2:#dc2626; --comment:#26364f; --commentbg:#f8fbff; --yellow:#b7791f; --code-bg:#fbfdff; --sidebar-bg:#e9eff7; --sidebar-bg-2:#eef4fa; --sidebar-panel:#f8fbff; --sidebar-panel-strong:#ffffff; --sidebar-border:#c7d4e3; --sidebar-text:#24324a; --shadow:0 1px 2px rgba(15,23,42,.035); --sticky-shadow:0 2px 6px rgba(15,23,42,.045); --app-header-height:86px; --sticky-top:0px; --comment-sticky-top:12px; }}
* {{ box-sizing:border-box; }}
html {{ height:100%; scroll-behavior:smooth; }}
body {{ margin:0; min-height:100vh; height:100vh; overflow:hidden; background:var(--bg); color:var(--text); display:grid; grid-template-rows:auto minmax(0,1fr); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ padding:10px 24px 0; border-bottom:1px solid var(--border); background:rgba(255,255,255,.94); backdrop-filter:saturate(160%) blur(12px); position:sticky; top:0; z-index:10; box-shadow:0 1px 0 rgba(15,23,42,.04); }}
h1 {{ margin:0; font-size:19px; line-height:1.25; letter-spacing:0; }}
a {{ color:var(--blue); }}
.header-top {{ display:flex; align-items:center; justify-content:space-between; gap:20px; }}
.brand {{ display:flex; align-items:center; min-width:0; gap:10px; }}
.brand-mark {{ width:34px; height:34px; display:grid; place-items:center; flex:0 0 auto; border-radius:8px; background:linear-gradient(135deg,var(--blue),var(--violet)); color:#fff; font-weight:800; font-size:12px; box-shadow:0 6px 14px rgba(37,99,235,.18); }}
.brand-copy {{ min-width:0; }}
.header-title {{ min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--title); }}
.header-subtitle {{ margin-top:1px; color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.meta {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; color:var(--muted); }}
.pill {{ border:1px solid var(--border); background:var(--surface2); border-radius:999px; padding:4px 10px; color:var(--muted); text-decoration:none; box-shadow:0 1px 0 rgba(15,23,42,.02); }}
.top-tabs {{ display:flex; justify-content:center; gap:6px; margin-top:8px; overflow-x:auto; }}
.top-tab {{ display:inline-flex; align-items:center; min-height:30px; padding:0 10px; border:1px solid transparent; border-bottom:2px solid transparent; border-radius:8px 8px 0 0; background:transparent; color:var(--muted); box-shadow:none; font:inherit; font-weight:650; white-space:nowrap; }}
.top-tab:hover,.top-tab:focus {{ color:var(--blue); background:var(--blue-soft); outline:none; }}
.top-tab.active {{ color:var(--blue); background:#fff; border-color:var(--border); border-bottom-color:var(--blue); outline:none; }}
.tab-shell {{ min-height:0; min-width:0; width:100%; max-width:100vw; height:100%; overflow:hidden; }}
.tab-panel {{ width:100%; max-width:100%; box-sizing:border-box; height:100%; min-height:0; overflow:auto; padding:18px 22px 80px; }}
.tab-panel[hidden] {{ display:none!important; }}
.files-tab {{ padding:0; overflow:hidden; }}
.layout {{ display:grid; grid-template-columns:320px minmax(0,1fr); min-height:0; height:100%; }}
.layout.sidebar-collapsed {{ grid-template-columns:minmax(0,1fr); }}
.sidebar-shell {{ border-right:1px solid var(--sidebar-border); background:linear-gradient(180deg,var(--sidebar-bg),var(--sidebar-bg-2)); position:static; height:100%; min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr); z-index:8; color:var(--sidebar-text); box-shadow:inset -1px 0 0 rgba(255,255,255,.65); }}
.layout.sidebar-collapsed .sidebar-shell {{ position:fixed; top:var(--app-header-height); left:0; width:auto; height:0; border-right:0; background:transparent; overflow:visible; }}
.sidebar-top {{ padding:12px 14px 10px; }}
.sidebar-title-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.sidebar-title-copy {{ min-width:0; }}
.sidebar-eyebrow {{ color:var(--text); font-weight:750; font-size:14px; line-height:1.2; }}
.sidebar-summary {{ margin-top:2px; color:#5c6f8a; font-size:12px; }}
nav {{ min-height:0; overflow:auto; padding:0 14px 16px; display:flex; flex-direction:column; gap:12px; }}
.layout.sidebar-collapsed nav {{ display:none; }}
main {{ grid-column:2; min-width:0; height:100%; overflow:auto; padding:0 22px 80px; }}
.layout.sidebar-collapsed main {{ grid-column:1; padding-left:58px; }}
input[type=search] {{ width:100%; padding:9px 10px; border-radius:7px; border:1px solid var(--sidebar-border); background:var(--surface); color:var(--text); margin:0; }}
input[type=search]:focus {{ outline:2px solid var(--blue-soft); border-color:var(--blue); }}
button {{ border:1px solid var(--border-strong); background:var(--surface); color:var(--text); padding:7px 10px; border-radius:6px; cursor:pointer; box-shadow:0 1px 1px rgba(15,23,42,.04); }}
button:hover {{ border-color:var(--blue); color:var(--blue); }}
.sidebar-toggle {{ width:32px; height:32px; flex:0 0 auto; display:grid; place-items:center; padding:0; border-color:var(--sidebar-border); background:var(--sidebar-panel-strong); font-weight:700; }}
.sidebar-toggle-icon {{ color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.sidebar-toggle-label {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
.layout.sidebar-collapsed .sidebar-top {{ padding:0; }}
.layout.sidebar-collapsed .sidebar-title-copy {{ display:none; }}
.layout.sidebar-collapsed .sidebar-toggle {{ width:38px; height:38px; margin:8px; padding:7px; justify-content:center; border-radius:999px; background:rgba(255,255,255,.96); box-shadow:0 10px 24px rgba(15,23,42,.14); }}
.layout.sidebar-collapsed .sidebar-toggle-label {{ display:none; }}
.sidebar-card,.sidebar-file-section {{ border:1px solid var(--sidebar-border); background:rgba(248,251,255,.88); border-radius:8px; padding:10px; box-shadow:0 1px 2px rgba(15,23,42,.035); }}
.sidebar-file-section {{ padding:10px 8px 8px; }}
.search-wrap {{ display:block; margin-bottom:8px; }}
.nav-actions {{ display:flex; gap:8px; }}
.nav-actions button {{ flex:1; }}
.nav-table-head {{ display:grid; grid-template-columns:minmax(0,1fr) 58px 70px; gap:8px; align-items:center; margin:0 0 6px; padding:0 6px; color:#6b7d95; font-size:10px; font-weight:750; text-transform:uppercase; letter-spacing:.02em; }}
.navfile {{ display:grid; grid-template-columns:minmax(0,1fr) 58px 70px; gap:8px; align-items:center; text-decoration:none; padding:8px 6px; border:1px solid transparent; border-radius:7px; color:var(--sidebar-text); overflow-wrap:anywhere; }}
.navfile:hover,.navfile:focus {{ background:rgba(255,255,255,.84); border-color:var(--sidebar-border); outline:none; }}
.navfile.reviewed {{ opacity:.8; }}
.navfile.reviewed .navfile-path {{ text-decoration:line-through; }}
.navfile-main {{ min-width:0; }}
.navfile-path {{ display:block; font-size:12px; line-height:1.25; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.navfile small {{ display:block; color:#6b7d95; margin-top:4px; font-size:11px; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.review-mark {{ display:none; }}
.nav-delta {{ justify-self:end; white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px; }}
.nav-status {{ justify-self:end; max-width:72px; overflow:hidden; text-overflow:ellipsis; border:1px solid #fed7aa; background:#fff7ed; color:#b45309; border-radius:999px; padding:2px 7px; font-size:11px; white-space:nowrap; }}
.navfile.reviewed .nav-status {{ border-color:#bbf7d0; background:#f0fdf4; color:var(--green2); }}
.overview {{ border:1px solid var(--border); background:var(--surface); border-radius:8px; margin-bottom:18px; overflow:hidden; box-shadow:var(--shadow); scroll-margin-top:16px; }}
.file {{ border:1px solid var(--border); background:var(--surface); border-radius:8px; margin:18px 0 0; overflow:visible; box-shadow:var(--shadow); scroll-margin-top:0; }}
.file.collapsed {{ overflow:hidden; box-shadow:var(--shadow); }}
.overview-body {{ padding:16px 18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.card {{ background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px; }}
.card h3 {{ margin:0 0 8px; font-size:14px; color:var(--text); }}
.card p,.card li {{ color:var(--muted); margin:0; }}
.architecture-intro {{ color:var(--muted); margin:0 0 14px; max-width:980px; }}
.architecture .overview-body,.arch-section {{ min-width:0; max-width:100%; }}
.arch-section {{ margin-top:16px; }}
.arch-section-head {{ margin:0 0 6px; }}
.arch-section h3 {{ margin:0; font-size:15px; color:var(--text); }}
.arch-section-summary {{ color:var(--muted); margin:0 0 10px; }}
.arch-flow-panel {{ position:relative; width:100%; max-width:100%; min-width:0; border:1px solid var(--border); border-radius:8px; background:linear-gradient(#ffffff,#f9fbff); box-sizing:border-box; overflow:hidden; }}
.flow-zoom-controls {{ position:absolute; top:10px; right:10px; z-index:8; display:flex; align-items:center; gap:4px; padding:4px; color:var(--muted); font-size:11px; border:1px solid var(--border); border-radius:8px; background:rgba(255,255,255,.92); box-shadow:0 2px 8px rgba(15,23,42,.08); backdrop-filter:blur(8px); }}
.flow-zoom-controls button {{ width:28px; height:28px; padding:0; display:inline-grid; place-items:center; border-radius:6px; font-weight:800; line-height:1; }}
.flow-zoom-level {{ min-width:42px; text-align:center; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.flow-focus-bar {{ display:none; align-items:center; justify-content:space-between; gap:10px; margin:12px 0 0; padding:10px 12px; border:1px solid var(--border); border-radius:8px; background:var(--blue-soft); color:#1e3a8a; }}
.flow-focus-bar.active {{ display:flex; }}
.arch-flow-scroll {{ width:100%; max-width:100%; min-width:0; overflow:auto; box-sizing:border-box; }}
.flow-zoom-stage {{ position:relative; min-width:100%; min-height:160px; }}
.flow-grid {{ position:absolute; top:0; left:0; display:grid; grid-template-columns:repeat(var(--flow-cols), 220px); gap:84px 148px; width:calc(var(--flow-cols) * 365px); max-width:none; padding:40px 34px 72px; box-sizing:border-box; transform-origin:top left; }}
.flow-svg {{ position:absolute; inset:0; width:100%; height:100%; z-index:0; pointer-events:none; overflow:visible; }}
.flow-edge-leader {{ stroke:#2563eb; stroke-width:1.2; stroke-dasharray:3 3; opacity:.62; fill:none; }}
.flow-edge-anchor {{ fill:#2563eb; opacity:.74; }}
.flow-edge-label-layer {{ position:absolute; inset:0; z-index:2; pointer-events:none; }}
.flow-edge-label {{ position:absolute; transform:translate(-50%,-50%); max-width:140px; padding:3px 8px; border:1px solid #bfdbfe; border-radius:999px; background:rgba(255,255,255,.96); color:#1d4ed8; font-size:11px; font-weight:700; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; box-shadow:0 3px 10px rgba(37,99,235,.14); pointer-events:auto; }}
.flow-edge-label:hover,.flow-edge-label:focus,.flow-edge-label.expanded {{ max-width:280px; white-space:normal; overflow:visible; z-index:6; outline:2px solid var(--blue-soft); }}
.flow-edge-detail {{ display:none; margin-top:8px; padding:10px 12px; border:1px solid #bfdbfe; border-radius:8px; background:var(--blue-soft); color:#1e3a8a; }}
.flow-edge-detail.active {{ display:block; }}
.flow-edge-detail strong {{ display:block; margin-bottom:4px; }}
.flow-edge-detail pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.flow-node {{ position:relative; z-index:1; display:block; width:100%; min-height:108px; padding:12px; text-align:left; white-space:normal; overflow-wrap:anywhere; background:var(--surface); border:1px solid var(--border-strong); border-radius:8px; color:var(--text); box-shadow:0 4px 12px rgba(15,23,42,.05); }}
.flow-node:hover {{ border-color:var(--blue); }}
.flow-node-title {{ font-weight:700; margin-bottom:6px; color:var(--text); overflow-wrap:anywhere; line-height:1.25; }}
.flow-node-detail {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
.flow-node-when {{ margin-top:8px; padding-top:8px; border-top:1px solid var(--border); color:var(--comment); font-size:11px; }}
.flow-node-when strong {{ color:var(--yellow); }}
.flow-node-evidence {{ margin-top:8px; color:var(--muted); font-size:11px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.flow-node-evidence strong {{ color:var(--yellow); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.evidence-line {{ margin-top:3px; overflow-wrap:anywhere; }}
.flow-node-files {{ margin-top:8px; color:var(--yellow); font-size:11px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.flow-kind-decision {{ border-color:#f6c453; background:#fffaf0; }}
.flow-kind-data,.flow-kind-storage {{ border-color:#93c5fd; background:#eff6ff; }}
.flow-kind-external {{ border-color:#c4b5fd; background:#f5f3ff; }}
.flow-kind-result {{ border-color:#86efac; background:#f0fdf4; }}
.glossary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }}
.glossary-item {{ background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px; }}
.glossary-term {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--blue); font-weight:700; margin-bottom:6px; overflow-wrap:anywhere; }}
.glossary-description {{ color:var(--muted); margin-bottom:8px; }}
.glossary-when {{ color:var(--comment); margin-bottom:8px; font-size:12px; }}
.glossary-when strong {{ color:var(--yellow); }}
.glossary-evidence {{ color:var(--muted); margin-bottom:8px; font-size:11px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.related-files-button {{ font-size:11px; padding:4px 7px; }}
.file-sticky {{ position:sticky; top:var(--sticky-top); z-index:6; background:var(--surface); border-radius:8px 8px 0 0; box-shadow:var(--sticky-shadow); }}
.file.collapsed .file-sticky {{ border-radius:8px; box-shadow:none; }}
.file-header {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px; align-items:center; padding:12px 14px; background:var(--surface); border-bottom:1px solid var(--border); border-radius:8px 8px 0 0; }}
.file-header[data-file-header-toggle] {{ cursor:pointer; }}
.file-header[data-file-header-toggle] button,.file-header[data-file-header-toggle] a,.file-header[data-file-header-toggle] input {{ cursor:default; }}
.file.collapsed .file-header {{ border-bottom:0; border-radius:8px; }}
.file-heading {{ display:grid; grid-template-columns:auto minmax(0,1fr); gap:10px; align-items:center; min-width:0; }}
.file-title-block {{ min-width:0; }}
.file-title {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow-wrap:anywhere; font-weight:700; }}
.file-subtitle {{ color:var(--muted); margin-top:3px; }}
.file-actions {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
.file-expander {{ width:28px; height:28px; padding:0; display:inline-grid; place-items:center; border:1px solid transparent; border-radius:6px; background:transparent; color:var(--muted); box-shadow:none; }}
.file-expander:hover,.file-expander:focus {{ background:var(--surface3); border-color:var(--border-strong); color:var(--blue); outline:none; }}
.file-expander-icon {{ width:16px; height:16px; transition:transform .12s ease; }}
.file:not(.collapsed) .file-expander-icon {{ transform:rotate(90deg); }}
.file.reviewed .file-header {{ background:#f0fdf4; }}
.review-status {{ border:1px solid #fed7aa; background:#fff7ed; border-radius:999px; padding:3px 8px; color:#b45309; font-size:12px; }}
.review-status.reviewed,.review-toggle.reviewed {{ border-color:#86efac; background:#f0fdf4; color:var(--green2); }}
.review-progress {{ position:relative; color:#43546e; font-size:12px; font-weight:650; margin:0; padding-bottom:12px; }}
.review-progress::after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:5px; border-radius:999px; background:#dce6f2; }}
.review-progress::before {{ content:""; position:absolute; left:0; bottom:0; z-index:1; width:var(--progress,0%); height:5px; border-radius:999px; background:linear-gradient(90deg,var(--blue),var(--violet)); }}
.review-tools {{ display:flex; flex-direction:column; gap:10px; margin:0; }}
.review-stat-row {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.review-stat {{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:6px 8px; border:1px solid var(--sidebar-border); border-radius:7px; background:rgba(255,255,255,.72); color:#5c6f8a; font-size:11px; }}
.review-stat strong {{ color:var(--text); font-size:13px; }}
.review-stat.pending strong {{ color:#b45309; }}
.review-stat.reviewed strong {{ color:var(--green2); }}
.comment-bulk-actions {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; }}
.comment-bulk-actions button {{ min-width:0; }}
.pending-filter {{ display:flex; gap:7px; align-items:center; color:#536782; font-size:12px; }}
.file-note {{ padding:12px 14px; color:var(--comment); background:var(--commentbg); border-bottom:1px solid var(--border); }}
.file.collapsed .file-note {{ display:none; }}
.hunk {{ display:grid; grid-template-columns:minmax(0,1fr) 380px; border-top:1px solid var(--border); overflow:hidden; }}
.hunk-code {{ min-width:0; max-width:100%; overflow-x:auto; overflow-y:hidden; }}
.hunk-comment {{ border-left:1px solid var(--border); background:var(--surface2); padding:12px; }}
.sticky {{ position:sticky; top:var(--comment-sticky-top); }}
.hunk-head {{ background:var(--surface3); color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; padding:7px 10px; border-bottom:1px solid var(--border); position:sticky; left:0; z-index:2; }}
.hunk-head-main {{ color:var(--text); font-weight:700; }}
.hunk-head-sub {{ color:var(--muted); margin-top:3px; font-size:11px; }}
.context-toolbar {{ display:flex; flex-wrap:wrap; gap:6px; padding:7px 10px; background:var(--surface); border-bottom:1px solid var(--border); position:sticky; left:0; z-index:2; }}
.context-button {{ font-size:11px; padding:3px 7px; }}
.selection-ref {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--yellow); overflow-wrap:anywhere; }}
table.diff .review-composer-row > td {{ position:sticky; left:0; z-index:4; width:var(--review-inline-width,100%); min-width:var(--review-inline-width,100%); max-width:var(--review-inline-width,100%); padding:0; border-top:1px solid var(--border); background:#fff8c5; white-space:normal; }}
.review-composer {{ position:relative; z-index:3; width:var(--review-inline-width,100%); max-width:var(--review-inline-width,100%); margin:0; padding:12px; border-left:3px solid var(--blue); background:#fff8c5; color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; overflow:hidden; }}
.review-composer-title {{ margin:0 0 8px; font-weight:700; color:var(--text); }}
.review-composer textarea {{ display:block; width:100%; min-height:112px; resize:vertical; padding:10px; border:1px solid var(--border-strong); border-radius:6px; background:var(--surface); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.review-composer textarea:focus {{ outline:2px solid var(--blue-soft); border-color:var(--blue); }}
.review-composer-actions {{ display:flex; justify-content:flex-end; gap:8px; margin-top:8px; }}
.review-composer-actions button {{ white-space:nowrap; }}
table.diff .review-comment-row > td {{ position:sticky; left:0; z-index:3; width:var(--review-inline-width,100%); min-width:var(--review-inline-width,100%); max-width:var(--review-inline-width,100%); padding:0; border-top:1px solid var(--border); background:#fffdf7; white-space:normal; }}
.review-comment-card {{ width:var(--review-inline-width,100%); max-width:var(--review-inline-width,100%); margin:0; padding:10px 12px; border-left:3px solid var(--blue); background:#fffdf7; color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; overflow:hidden; }}
.review-comment-meta {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:6px; color:var(--muted); }}
.review-comment-ref {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--yellow); overflow-wrap:anywhere; }}
.review-comment-actions {{ display:flex; gap:6px; flex-wrap:wrap; }}
.review-comment-actions button {{ padding:4px 8px; font-size:12px; }}
.review-comment-body {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--comment); }}
table.diff {{ border-collapse:collapse; width:max-content; min-width:100%; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; table-layout:auto; }}
.diff td {{ vertical-align:top; white-space:pre; }}
.lno {{ width:48px; color:var(--muted2); text-align:right; padding:0 8px; user-select:none; border-right:1px solid rgba(15,23,42,.06); }}
.sign {{ width:20px; text-align:center; color:var(--muted); user-select:none; }}
.code {{ color:#1f2937; padding-right:10px; min-width:60ch; }}
.code code {{ display:inline; padding:0; background:transparent; color:inherit; font:inherit; white-space:pre; }}
.code code.hljs {{ display:inline; padding:0; background:transparent; color:inherit; overflow:visible; }}
.code .hljs-keyword,.code .hljs-built_in,.code .hljs-selector-tag {{ font-weight:600; }}
tr.selectable {{ cursor:pointer; }}
tr.selected {{ outline:1px solid var(--yellow); outline-offset:-1px; box-shadow:inset 0 0 0 9999px rgba(251,191,36,.18); }}
tr.selected .lno,tr.selected .sign {{ color:#92400e; }}
tr.commented {{ box-shadow:inset 3px 0 0 var(--blue); }}
tr.add {{ background:var(--green); }} tr.add .sign,tr.add .lno {{ background:#dcfce7; color:var(--green2); }}
tr.del {{ background:var(--red); }} tr.del .sign,tr.del .lno {{ background:#ffe4e6; color:var(--red2); }}
tr.ctx {{ background:var(--code-bg); }} tr.extra {{ background:#f8fafc; color:var(--muted); }} tr.meta {{ background:var(--surface3); color:var(--muted); }}
.comment-title {{ color:var(--text); font-weight:700; margin-bottom:8px; }}
.comment-list {{ margin:8px 0 0; padding-left:18px; }}
.comment-list li {{ margin:0 0 8px; color:var(--comment); }}
.symbols {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }}
.symbol {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px; border:1px solid var(--border); background:var(--surface); color:var(--yellow); padding:2px 6px; border-radius:999px; }}
.badge {{ display:inline-flex; gap:5px; align-items:center; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
.file-delta {{ white-space:nowrap; }}
.adds {{ color:var(--green2); }} .dels {{ color:var(--red2); }} .hidden {{ display:none!important; }}
@media(max-width:1100px) {{ body {{ overflow:hidden; }} .tab-panel {{ padding:18px 22px 80px; }} .files-tab {{ overflow:auto; padding:0; }} .layout,.layout.sidebar-collapsed {{ grid-template-columns:1fr; height:auto; min-height:100%; }} .sidebar-shell {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--border); }} .layout.sidebar-collapsed .sidebar-shell {{ position:sticky; top:0; height:auto; border-bottom:0; z-index:9; }} main,.layout.sidebar-collapsed main {{ grid-column:1; height:auto; overflow:visible; padding:0 22px 80px; }} nav {{ height:auto; max-height:52vh; }} .hunk {{ grid-template-columns:1fr; }} .hunk-comment {{ border-left:0; border-top:1px solid var(--border); }} .sticky {{ position:static; }} .top-tabs {{ justify-content:flex-start; gap:14px; }} .header-top {{ align-items:flex-start; flex-direction:column; gap:8px; }} .header-title {{ white-space:normal; }} .meta {{ justify-content:flex-start; }} }}
</style>
</head>
<body>
<header><div class="header-top"><div class="brand"><div class="brand-mark" aria-hidden="true">DW</div><div class="brand-copy"><h1 class="header-title">{html.escape(page_title)}</h1><div class="header-subtitle">{html.escape(model['title'])}</div></div></div><div class="meta"><span class="pill">{html.escape(str(model['range']))}</span><span class="pill">head {html.escape(model['shortHead'])}</span><span class="pill">{html.escape(ui['grounding'].lower())} {html.escape(model['generatedAt'])}</span>{'<a class="pill" href="' + html.escape(model['pr']['url']) + '">GitHub PR</a>' if model.get('pr') and model['pr'].get('url') else ''}</div></div><div class="top-tabs" role="tablist" aria-label="Walkthrough sections"><button class="top-tab active" type="button" role="tab" aria-selected="true" aria-controls="panel-summary" data-tab-target="summary">{html.escape(ui['how_to_read'])}</button><button class="top-tab" type="button" role="tab" aria-selected="false" aria-controls="panel-architecture" data-tab-target="architecture">{html.escape(ui['architecture'])}</button><button class="top-tab" type="button" role="tab" aria-selected="false" aria-controls="panel-files" data-tab-target="files">{html.escape(ui['files_tab'])}</button></div></header>
<div class="tab-shell" id="tabShell"><section class="tab-panel active" id="panel-summary" data-tab-panel="summary" role="tabpanel"></section><section class="tab-panel" id="panel-architecture" data-tab-panel="architecture" role="tabpanel" hidden></section><section class="tab-panel files-tab" id="panel-files" data-tab-panel="files" role="tabpanel" hidden><div class="layout" id="layout"><aside class="sidebar-shell"><div class="sidebar-top"><div class="sidebar-title-row"><div class="sidebar-title-copy"><div class="sidebar-eyebrow">{html.escape(ui['review_queue'])}</div><div class="sidebar-summary"><span id="reviewQueueSummary">0 / {len(model['files'])} {html.escape(ui['reviewed_progress_suffix'])}</span></div></div><button id="sidebarToggle" class="sidebar-toggle" aria-expanded="true" aria-controls="sidebarNav" title="{html.escape(ui['hide_sidebar'])}"><span class="sidebar-toggle-label">{html.escape(ui['hide_sidebar'])}</span><span class="sidebar-toggle-icon" aria-hidden="true">&lt;&lt;</span></button></div></div><nav id="sidebarNav"><section class="sidebar-card sidebar-search-card"><div class="search-wrap"><input id="filter" type="search" placeholder="{html.escape(ui['filter'])}"></div><div class="nav-actions"><button id="expandAll">{html.escape(ui['expand_all'])}</button><button id="collapseAll">{html.escape(ui['collapse_all'])}</button></div></section><section class="sidebar-card review-tools"><div id="reviewProgress" class="review-progress"></div><div class="review-stat-row"><div class="review-stat pending"><span>{html.escape(ui['pending'])}</span><strong id="pendingCount">0</strong></div><div class="review-stat reviewed"><span>{html.escape(ui['reviewed'])}</span><strong id="reviewedCount">0</strong></div></div><label class="pending-filter"><input id="pendingOnly" type="checkbox"> {html.escape(ui['show_pending_only'])}</label><div class="comment-bulk-actions"><button id="copyAllComments">{html.escape(ui['copy_all_comments'])}</button><button id="clearAllComments" title="{html.escape(ui['clear_all_comments_title'])}">{html.escape(ui['clear_all_comments'])}</button></div><button id="clearReviewed">{html.escape(ui['clear_reviewed'])}</button></section><section class="sidebar-file-section"><div class="nav-table-head"><span>{html.escape(ui['file'])}</span><span>{html.escape(ui['changes'])}</span><span>{html.escape(ui['status'])}</span></div><div id="navFiles"></div></section></nav></aside><main id="app"></main></div></section></div><div id="reviewComposer" class="review-composer hidden" role="dialog" aria-live="polite"><div class="review-composer-title" id="reviewComposerTitle"></div><textarea id="reviewComposerText" placeholder="{html.escape(ui['comment_placeholder'])}"></textarea><div class="review-composer-actions"><button id="commentCancel">{html.escape(ui['cancel'])}</button><button id="commentSave">{html.escape(ui['save'])}</button></div></div>
<script id="walkthrough-data" type="application/json">{json_payload}</script>
<script>
const data=JSON.parse(document.getElementById('walkthrough-data').textContent); const ui=data.ui;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const app=document.getElementById('app'), summaryPanel=document.getElementById('panel-summary'), architecturePanel=document.getElementById('panel-architecture'), navFiles=document.getElementById('navFiles'), pageHeader=document.querySelector('body > header'), layout=document.getElementById('layout'), sidebarToggle=document.getElementById('sidebarToggle'), sidebarNav=document.getElementById('sidebarNav'), reviewComposer=document.getElementById('reviewComposer'), reviewComposerTitle=document.getElementById('reviewComposerTitle'), reviewComposerText=document.getElementById('reviewComposerText'), copyAllCommentsButton=document.getElementById('copyAllComments'), clearAllCommentsButton=document.getElementById('clearAllComments'), reviewQueueSummary=document.getElementById('reviewQueueSummary'), reviewedCountLabel=document.getElementById('reviewedCount'), pendingCountLabel=document.getElementById('pendingCount');
let selection=null;
let showPendingOnly=false;
let architectureFocus=null;
let activeTab='summary';
let editingCommentId=null;
let composerRow=null;
const tabScrollPositions={{summary:0,architecture:0,files:0}};
const FLOW_ZOOM_MIN=0.6, FLOW_ZOOM_MAX=1.5, FLOW_ZOOM_STEP=0.15;
const reviewStorageKey=`code-diff-walkthrough:reviewed:v1:${{data.reviewStateKey}}`;
const sidebarStorageKey=`code-diff-walkthrough:sidebar:v1:${{data.reviewStateKey}}`;
const expansionStorageKey=`code-diff-walkthrough:expanded:v1:${{data.reviewStateKey}}`;
const commentsStorageKey=`code-diff-walkthrough:comments:v1:${{data.reviewStateKey}}`;
const flowZoomStorageKey=`code-diff-walkthrough:flow-zoom:v1:${{data.reviewStateKey}}`;
let reviewState=loadReviewState();
let sidebarCollapsed=loadSidebarState();
let expansionState=loadExpansionState();
let commentsState=loadCommentsState();
let flowZoomState=loadFlowZoomState();
function loadReviewState(){{try{{const raw=localStorage.getItem(reviewStorageKey); const parsed=raw?JSON.parse(raw):null; if(parsed&&typeof parsed==='object'&&parsed.files&&typeof parsed.files==='object')return parsed;}}catch(e){{}} return {{files:{{}}}};}}
function saveReviewState(){{try{{localStorage.setItem(reviewStorageKey,JSON.stringify(reviewState));}}catch(e){{}}}}
function loadSidebarState(){{try{{return localStorage.getItem(sidebarStorageKey)==='collapsed';}}catch(e){{return false;}}}}
function saveSidebarState(){{try{{localStorage.setItem(sidebarStorageKey,sidebarCollapsed?'collapsed':'expanded');}}catch(e){{}}}}
function loadExpansionState(){{try{{const raw=localStorage.getItem(expansionStorageKey); const parsed=raw?JSON.parse(raw):null; if(parsed&&typeof parsed==='object'&&parsed.files&&typeof parsed.files==='object')return parsed;}}catch(e){{}} return {{files:{{}}}};}}
function saveExpansionState(){{try{{localStorage.setItem(expansionStorageKey,JSON.stringify(expansionState));}}catch(e){{}}}}
function loadCommentsState(){{try{{const raw=localStorage.getItem(commentsStorageKey); const parsed=raw?JSON.parse(raw):null; if(parsed&&typeof parsed==='object'&&Array.isArray(parsed.comments))return parsed;}}catch(e){{}} return {{comments:[]}};}}
function saveCommentsState(){{try{{localStorage.setItem(commentsStorageKey,JSON.stringify(commentsState));}}catch(e){{}}}}
function loadFlowZoomState(){{try{{const raw=localStorage.getItem(flowZoomStorageKey); const parsed=raw?JSON.parse(raw):null; if(parsed&&typeof parsed==='object'&&parsed.flows&&typeof parsed.flows==='object')return parsed;}}catch(e){{}} return {{flows:{{}}}};}}
function saveFlowZoomState(){{try{{localStorage.setItem(flowZoomStorageKey,JSON.stringify(flowZoomState));}}catch(e){{}}}}
function setSidebarCollapsed(value){{sidebarCollapsed=Boolean(value); layout?.classList.toggle('sidebar-collapsed',sidebarCollapsed); sidebarNav?.setAttribute('aria-hidden',sidebarCollapsed?'true':'false'); if(sidebarToggle){{sidebarToggle.setAttribute('aria-expanded',String(!sidebarCollapsed)); sidebarToggle.title=sidebarCollapsed?ui.show_sidebar:ui.hide_sidebar; const label=sidebarToggle.querySelector('.sidebar-toggle-label'); if(label)label.textContent=sidebarCollapsed?ui.show_sidebar:ui.hide_sidebar; const icon=sidebarToggle.querySelector('.sidebar-toggle-icon'); if(icon)icon.textContent=sidebarCollapsed?'>>':'<<';}} updateStickyOffsets(); if(activeTab==='architecture')drawArchitectureEdges();}}
function clampFlowZoom(value){{return Math.min(FLOW_ZOOM_MAX,Math.max(FLOW_ZOOM_MIN,Math.round(Number(value||1)*100)/100));}}
function flowZoomFor(grid){{const id=grid?.dataset.flowId; const stored=id?Number(flowZoomState.flows?.[id]):NaN; return clampFlowZoom(Number.isFinite(stored)&&stored>0?stored:Number(grid?.dataset.flowCurrentZoom||1));}}
function updateFlowZoomControls(grid){{const zoom=flowZoomFor(grid); const section=grid?.closest('.arch-section'); const level=section?.querySelector('[data-flow-zoom-level]'); if(level)level.textContent=`${{Math.round(zoom*100)}}%`; section?.querySelectorAll('button[data-flow-zoom]').forEach(button=>{{const direction=Number(button.dataset.flowZoom); button.disabled=direction<0?zoom<=FLOW_ZOOM_MIN+.001:zoom>=FLOW_ZOOM_MAX-.001;}});}}
function syncFlowZoomStage(grid,zoom){{const stage=grid?.closest('.flow-zoom-stage'); if(!grid||!stage)return; const width=grid.offsetWidth, height=grid.offsetHeight; if(width>0&&height>0){{stage.style.width=`${{Math.ceil(width*zoom)}}px`; stage.style.height=`${{Math.ceil(height*zoom)}}px`;}}}}
function applyFlowZoom(grid,zoom){{if(!grid)return; zoom=clampFlowZoom(zoom); delete grid.dataset.flowZoom; grid.dataset.flowCurrentZoom=String(zoom); grid.style.transform=`scale(${{zoom}})`; syncFlowZoomStage(grid,zoom); updateFlowZoomControls(grid);}}
function setStoredFlowZoom(grid,zoom){{if(!grid?.dataset.flowId)return; flowZoomState.flows=flowZoomState.flows||{{}}; const next=clampFlowZoom(zoom); if(Math.abs(next-1)<.001)delete flowZoomState.flows[grid.dataset.flowId]; else flowZoomState.flows[grid.dataset.flowId]=next; saveFlowZoomState(); applyFlowZoom(grid,next);}}
function initializeFlowZooms(){{document.querySelectorAll('.flow-grid').forEach(grid=>applyFlowZoom(grid,flowZoomFor(grid)));}}
function bindFlowZoomControls(){{document.querySelectorAll('button[data-flow-zoom]').forEach(button=>button.addEventListener('click',()=>{{const section=button.closest('.arch-section'); const grid=section?.querySelector('.flow-grid'); const scroller=section?.querySelector('.arch-flow-scroll'); if(!grid||!scroller)return; const centerX=(scroller.scrollLeft+scroller.clientWidth/2)/Math.max(1,scroller.scrollWidth); const centerY=(scroller.scrollTop+scroller.clientHeight/2)/Math.max(1,scroller.scrollHeight); const next=clampFlowZoom(flowZoomFor(grid)+Number(button.dataset.flowZoom)*FLOW_ZOOM_STEP); setStoredFlowZoom(grid,next); requestAnimationFrame(()=>{{drawArchitectureEdges(); scroller.scrollLeft=Math.max(0,centerX*scroller.scrollWidth-scroller.clientWidth/2); scroller.scrollTop=Math.max(0,centerY*scroller.scrollHeight-scroller.clientHeight/2);}});}}));}}
function fileByPath(path){{return data.files.find(f=>f.new_path===path);}}
function isReviewed(path){{const f=fileByPath(path), entry=reviewState.files?.[path]; return Boolean(f&&entry&&entry.fingerprint===f.reviewFingerprint);}}
function setReviewed(path,value){{const f=fileByPath(path); if(!f)return; reviewState.files=reviewState.files||{{}}; if(value)reviewState.files[path]={{fingerprint:f.reviewFingerprint,reviewedAt:new Date().toISOString()}}; else delete reviewState.files[path]; saveReviewState(); updateReviewUi();}}
function isFileExpanded(path){{const f=fileByPath(path), entry=expansionState.files?.[path]; return Boolean(f&&entry&&entry.fingerprint===f.reviewFingerprint&&entry.expanded===true);}}
function setFileExpanded(path,value){{const f=fileByPath(path); if(!f)return; expansionState.files=expansionState.files||{{}}; if(value)expansionState.files[path]={{fingerprint:f.reviewFingerprint,expanded:true,updatedAt:new Date().toISOString()}}; else delete expansionState.files[path]; saveExpansionState();}}
function validReviewComments(){{return (commentsState.comments||[]).filter(comment=>{{const f=fileByPath(comment.file); return Boolean(f&&comment.fingerprint===f.reviewFingerprint&&Number.isFinite(comment.start)&&Number.isFinite(comment.end)&&String(comment.text||'').trim());}});}}
function clearReviewed(){{reviewState={{files:{{}}}}; saveReviewState(); updateReviewUi();}}
function updateStickyOffsets(){{const height=Math.ceil(pageHeader?.getBoundingClientRect().height||92); document.documentElement.style.setProperty('--app-header-height',`${{height}}px`);}}
function panelForTab(tab){{return document.querySelector(`[data-tab-panel="${{CSS.escape(tab)}}"]`);}}
function scrollContainerForTab(tab){{return tab==='files'?app:panelForTab(tab);}}
function saveActiveTabScroll(){{const container=scrollContainerForTab(activeTab); if(container)tabScrollPositions[activeTab]=container.scrollTop;}}
function setActiveTab(tab){{const next=panelForTab(tab); if(!next||(tab===activeTab&&!next.hidden))return; saveActiveTabScroll(); activeTab=tab; document.querySelectorAll('[data-tab-panel]').forEach(panel=>{{const selected=panel.dataset.tabPanel===tab; panel.hidden=!selected; panel.classList.toggle('active',selected);}}); document.querySelectorAll('[data-tab-target]').forEach(button=>{{const selected=button.dataset.tabTarget===tab; button.classList.toggle('active',selected); button.setAttribute('aria-selected',String(selected));}}); requestAnimationFrame(()=>{{const container=scrollContainerForTab(tab); if(container)container.scrollTop=tabScrollPositions[tab]||0; updateStickyOffsets(); if(tab==='architecture')drawArchitectureEdges(); if(tab==='files'){{autoFitAllHunkContext(); positionReviewComposer();}} else hideReviewComposer();}});}}
function bindTabs(){{document.querySelectorAll('[data-tab-target]').forEach(button=>button.addEventListener('click',()=>setActiveTab(button.dataset.tabTarget))); document.querySelectorAll('[data-tab-link]').forEach(link=>link.addEventListener('click',event=>{{event.preventDefault(); setActiveTab(link.dataset.tabLink);}}));}}
function bindFileNavigation(){{document.querySelectorAll('a.navfile[href^="#file-"]').forEach(link=>link.addEventListener('click',event=>{{event.preventDefault(); const targetId=link.getAttribute('href'); setActiveTab('files'); requestAnimationFrame(()=>document.querySelector(targetId)?.scrollIntoView({{block:'start'}}));}}));}}
function localize(value){{if(value&&typeof value==='object')return value[data.language]??value.en??value.es??''; return value??'';}}
function shortPath(path){{const parts=String(path||'').split('/').filter(Boolean); return parts.at(-1)||String(path||'');}}
const LANGUAGE_EXTENSIONS=new Map([['cs','csharp'],['csx','csharp'],['java','java'],['go','go'],['js','javascript'],['jsx','javascript'],['mjs','javascript'],['cjs','javascript'],['ts','typescript'],['tsx','typescript'],['py','python'],['rb','ruby'],['rs','rust'],['c','cpp'],['h','cpp'],['cc','cpp'],['cpp','cpp'],['cxx','cpp'],['hpp','cpp'],['fs','fsharp'],['fsx','fsharp'],['vb','vbnet'],['kt','kotlin'],['kts','kotlin'],['swift','swift'],['scala','scala'],['sh','bash'],['bash','bash'],['zsh','bash'],['ps1','powershell'],['sql','sql'],['json','json'],['yaml','yaml'],['yml','yaml'],['toml','ini'],['xml','xml'],['html','xml'],['htm','xml'],['csproj','xml'],['fsproj','xml'],['vbproj','xml'],['props','xml'],['targets','xml'],['config','xml'],['md','markdown'],['markdown','markdown'],['css','css'],['scss','css'],['less','css']]);
function languageForPath(path){{const clean=String(path||'').replace(/\\\\/g,'/'); const name=(clean.split('/').pop()||'').toLowerCase(); if(name==='dockerfile'||name.startsWith('dockerfile.'))return 'dockerfile'; if(name==='makefile'||name.endsWith('.mk'))return 'makefile'; const parts=name.split('.'); const ext=parts.length>1?parts.pop():''; return LANGUAGE_EXTENSIONS.get(ext)||'plaintext';}}
function applySyntaxHighlighting(root=document){{if(!window.hljs)return; root.querySelectorAll('td.code code:not([data-highlighted])').forEach(block=>{{try{{window.hljs.highlightElement(block);}}catch(e){{}}}});}}
function hasArchitecture(){{const a=data.architecture||{{}}; return Boolean((a.sections&&a.sections.length)||(a.glossary&&a.glossary.length));}}
function encodeFiles(files){{return esc(JSON.stringify(files||[]));}}
function decodeFiles(raw){{try{{const parsed=JSON.parse(raw||'[]'); return Array.isArray(parsed)?parsed:[];}}catch(e){{return [];}}}}
function focusArchitectureFiles(files){{if(!files||!files.length)return; architectureFocus=new Set(files); const filterInput=document.getElementById('filter'); if(filterInput)filterInput.value=''; updateArchitectureFocusUi(); applyFilters(); const first=data.files.find(f=>architectureFocus.has(f.new_path)); setActiveTab('files'); requestAnimationFrame(()=>{{const target=first?document.querySelector(`section.file[data-file-path="${{CSS.escape(first.new_path)}}"]`):null; if(target)target.scrollIntoView({{block:'start'}});}});}}
function clearArchitectureFocus(){{architectureFocus=null; updateArchitectureFocusUi(); applyFilters();}}
function updateArchitectureFocusUi(){{const active=Boolean(architectureFocus&&architectureFocus.size); const label=active?`${{ui.flow_focus}}: ${{Array.from(architectureFocus).join(', ')}}`:''; document.querySelectorAll('.flow-focus-bar').forEach(el=>el.classList.toggle('active',active)); document.querySelectorAll('.arch-focus-state').forEach(el=>el.textContent=label);}}
function reviewedCount(){{return data.files.reduce((n,f)=>n+(isReviewed(f.new_path)?1:0),0);}}
function line(l,filePath){{const sign=l.kind==='add'?'+':l.kind==='del'?'-':l.kind==='extra'?'·':l.kind==='ctx'?' ':''; const refLine=l.new??l.old??''; const selectable=Number.isFinite(refLine); const lang=languageForPath(filePath); return `<tr class="${{l.kind}}${{selectable?' selectable':''}}" data-file="${{esc(filePath)}}" data-line="${{esc(refLine)}}" title="${{selectable?esc(`${{filePath}}:${{refLine}}`):''}}"><td class="lno">${{l.old??''}}</td><td class="lno">${{l.new??''}}</td><td class="sign">${{sign}}</td><td class="code"><code class="language-${{esc(lang)}}">${{esc(l.text)}}</code></td></tr>`;}}
function contextLabel(h){{const c=h.context||{{}}; const range=c.newStart&&c.newEnd?`${{ui.new_lines}} ${{c.newStart}}-${{c.newEnd}}`:''; return [c.symbol, range].filter(Boolean).join(' · ') || h.header;}}
function extraRows(rows, cls, filePath, side){{return rows&&rows.length?`<tbody class="${{cls}} extra-context hidden" data-extra-id="${{esc(cls)}}" data-context-side="${{esc(side)}}">${{rows.map(l=>line(l,filePath)).join('')}}</tbody>`:'';}}
function contextToolbar(h, id){{const before=h.context?.before?.length||0, after=h.context?.after?.length||0; if(!before&&!after)return ''; const buttons=[]; if(before)buttons.push(`<button class="context-button" data-extra-target="${{id}}-before" data-extra-label="${{esc(ui.show_more_before)}}">${{esc(ui.show_more_before)}} (${{before}})</button>`); if(after)buttons.push(`<button class="context-button" data-extra-target="${{id}}-after" data-extra-label="${{esc(ui.show_more_after)}}">${{esc(ui.show_more_after)}} (${{after}})</button>`); return `<div class="context-toolbar">${{buttons.join('')}}</div>`;}}
function hunk(h, index, filePath){{const hid=`hunk-${{Math.random().toString(36).slice(2)}}-${{index}}`; const label=contextLabel(h); const notes=h.comment.notes[data.language].map(n=>`<li>${{esc(n)}}</li>`).join(''); const sy=h.comment.symbols.length?`<div class="comment-title" style="margin-top:12px">${{esc(ui.changed_blocks)}}</div><div class="symbols">${{h.comment.symbols.map(s=>`<span class="symbol">${{esc(s)}}</span>`).join('')}}</div>`:''; const search=(h.header+' '+label+' '+h.comment.notes[data.language].join(' ')+' '+h.comment.symbols.join(' ')).toLowerCase(); return `<div class="hunk" data-search="${{esc(search)}}"><div class="hunk-code"><div class="hunk-head"><div class="hunk-head-main">${{esc(label)}} · <span class="adds">+${{h.comment.adds}}</span> <span class="dels">-${{h.comment.dels}}</span></div><div class="hunk-head-sub">${{esc(ui.original_hunk)}}: ${{esc(h.header)}}</div></div>${{contextToolbar(h,hid)}}<table class="diff">${{extraRows(h.context?.before,`${{hid}}-before`,filePath,'before')}}<tbody>${{h.lines.map(l=>line(l,filePath)).join('')}}</tbody>${{extraRows(h.context?.after,`${{hid}}-after`,filePath,'after')}}</table></div><aside class="hunk-comment"><div class="sticky"><div class="comment-title">${{esc(ui.why_block)}}</div><div style="color:var(--muted); margin-bottom:8px">${{esc(ui.context)}}: ${{esc(label)}}</div><ul class="comment-list">${{notes}}</ul>${{sy}}</div></aside></div>`;}}
function evidenceLabel(item){{if(!item)return ''; const label=localize(item.label); if(label)return label; if(item.file&&item.line)return `${{item.file}}:${{item.line}}`; return item.file||'';}}
function evidenceBlock(items, className){{if(!items||!items.length)return ''; return `<div class="${{className}}"><strong>${{esc(ui.evidence)}}:</strong>${{items.map(item=>`<div class="evidence-line">${{esc(evidenceLabel(item))}}</div>`).join('')}}</div>`;}}
function architectureNode(node){{const files=node.files||[]; const detail=localize(node.detail); const when=localize(node.when); const whenBlock=when?`<div class="flow-node-when"><strong>${{esc(ui.when)}}:</strong> ${{esc(when)}}</div>`:''; const evidence=evidenceBlock(node.evidence,'flow-node-evidence'); const fileLine=files.length?`<div class="flow-node-files">${{files.length}} ${{esc(ui.related_files)}}</div>`:''; const kind=esc(node.kind||'process'); return `<button class="flow-node flow-kind-${{kind}}" style="grid-column:${{Number(node.column)||1}};grid-row:${{Number(node.row)||1}}" data-node-id="${{esc(node.id)}}" data-arch-files="${{encodeFiles(files)}}" title="${{files.map(String).join('\\n')}}"><div class="flow-node-title">${{esc(localize(node.label)||node.id)}}</div>${{detail?`<div class="flow-node-detail">${{esc(detail)}}</div>`:''}}${{whenBlock}}${{evidence}}${{fileLine}}</button>`;}}
function architectureFlow(section,index){{const nodes=section.nodes||[]; const edges=section.edges||[]; const summary=localize(section.summary); const flowId=`flow-${{index}}`; const columns=Number(section.columns)||Math.max(nodes.length,1); return `<div class="arch-section"><div class="arch-section-head"><h3>${{esc(localize(section.title))}}</h3></div>${{summary?`<p class="arch-section-summary">${{esc(summary)}}</p>`:''}}<div class="arch-flow-panel"><div class="flow-zoom-controls" data-flow-controls="${{flowId}}"><button type="button" data-flow-zoom="-1" title="${{esc(ui.zoom_out)}}" aria-label="${{esc(ui.zoom_out)}}">-</button><span class="flow-zoom-level" data-flow-zoom-level title="${{esc(ui.zoom_level)}}">100%</span><button type="button" data-flow-zoom="1" title="${{esc(ui.zoom_in)}}" aria-label="${{esc(ui.zoom_in)}}">+</button></div><div class="arch-flow-scroll"><div class="flow-zoom-stage"><div class="flow-grid" data-flow-id="${{flowId}}" data-flow-cols="${{columns}}" data-edges="${{esc(JSON.stringify(edges))}}" style="--flow-cols:${{columns}}"><svg class="flow-svg" aria-hidden="true"></svg><div class="flow-edge-label-layer"></div>${{nodes.map(architectureNode).join('')}}</div></div></div></div><div class="flow-edge-detail"></div></div>`;}}
function architectureGlossaryItem(item){{const files=item.files||[]; const when=localize(item.when); const whenBlock=when?`<div class="glossary-when"><strong>${{esc(ui.when)}}:</strong> ${{esc(when)}}</div>`:''; const evidence=evidenceBlock(item.evidence,'glossary-evidence'); const button=files.length?`<button class="related-files-button" data-arch-files="${{encodeFiles(files)}}">${{esc(ui.focus_related_files)}}</button>`:''; return `<div class="glossary-item"><div class="glossary-term">${{esc(item.term)}}</div><div class="glossary-description">${{esc(localize(item.description))}}</div>${{whenBlock}}${{evidence}}${{button}}</div>`;}}
function architecture(){{const arch=data.architecture||{{}}; const title=localize(arch.title)||ui.architecture; const summary=localize(arch.summary)||ui.architecture_text; const flows=(arch.sections||[]).map(architectureFlow).join(''); const glossary=(arch.glossary||[]).length?`<div class="arch-section" id="glossary"><h3>${{esc(ui.type_glossary)}}</h3><div class="glossary-grid">${{arch.glossary.map(architectureGlossaryItem).join('')}}</div></div>`:''; return `<section class="overview architecture" id="architecture"><div class="file-header"><strong>${{esc(title)}}</strong><span class="badge">${{esc(ui.architecture_flows)}}</span></div><div class="overview-body"><p class="architecture-intro">${{esc(summary)}}</p><div class="flow-focus-bar"><span class="arch-focus-state"></span><button class="arch-clear-focus">${{esc(ui.clear_flow_focus)}}</button></div>${{flows}}${{glossary}}</div></section>`;}}
function overview(){{const cats=Object.entries(data.categories[data.language]).map(([n,c])=>`<div class="card"><h3>${{esc(n)}}</h3><p>${{c.files}} ${{esc(ui.files)}} · <span class="adds">+${{c.add}}</span> <span class="dels">-${{c.del}}</span></p></div>`).join(''); const commits=data.commitLog.map(c=>`<li><code>${{esc(c)}}</code></li>`).join(''); return `<section class="overview" id="summary"><div class="file-header"><strong>${{esc(ui.how_to_read)}}</strong><span class="badge"><span class="adds">+${{data.files.reduce((n,f)=>n+Number(f.numstat.additions||0),0)}}</span> <span class="dels">-${{data.files.reduce((n,f)=>n+Number(f.numstat.deletions||0),0)}}</span></span></div><div class="overview-body"><div class="grid"><div class="card"><h3>${{esc(ui.purpose)}}</h3><p>${{esc(ui.purpose_text)}}</p></div><div class="card"><h3>${{esc(ui.grounding)}}</h3><p>${{esc(ui.grounding_text)}}</p></div><div class="card"><h3>${{esc(ui.navigation)}}</h3><p>${{esc(ui.navigation_text)}}</p></div></div><h3>${{esc(ui.change_areas)}}</h3><div class="grid">${{cats}}</div><h3>${{esc(ui.commit_story)}}</h3><ol>${{commits}}</ol></div></section>`;}}
function file(f,i){{const id=`file-${{i}}`; const search=`${{f.new_path}} ${{f.category[data.language]}} ${{f.note[data.language]}} ${{f.hunks.map(h=>contextLabel(h)+' '+h.comment.notes[data.language].join(' ')).join(' ')}}`.toLowerCase(); return `<section id="${{id}}" class="file collapsed" data-file-path="${{esc(f.new_path)}}" data-search="${{esc(search)}}"><div class="file-sticky"><div class="file-header" data-file-header-toggle="${{id}}-body" title="${{esc(ui.expand)}}"><div class="file-heading"><button class="toggle file-expander" data-target="${{id}}-body" title="${{esc(ui.expand)}}" aria-label="${{esc(ui.expand)}}" aria-expanded="false"><svg class="file-expander-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18l6-6-6-6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></path></svg></button><div class="file-title-block"><div class="file-title">${{esc(f.new_path)}}</div><div class="file-subtitle">${{esc(f.category[data.language])}} · ${{esc(f.status)}}</div></div></div><div class="file-actions"><span class="badge file-delta"><span class="adds">+${{esc(f.numstat.additions)}}</span> <span class="dels">-${{esc(f.numstat.deletions)}}</span></span><span class="review-status" data-reviewed-status="${{esc(f.new_path)}}">${{esc(ui.pending)}}</span><button class="review-toggle" data-reviewed-toggle="${{esc(f.new_path)}}">${{esc(ui.mark_reviewed)}}</button></div></div><div class="file-note"><strong>${{esc(ui.file_rationale)}}:</strong> ${{esc(f.note[data.language])}}</div></div><div id="${{id}}-body" class="file-body hidden">${{f.hunks.map((h,idx)=>hunk(h,idx,f.new_path)).join('')}}</div></section>`;}}
function contextRows(group){{return Array.from(group?.querySelectorAll('tr')||[]);}}
function hiddenContextRows(group){{return contextRows(group).filter(row=>row.classList.contains('hidden'));}}
function contextButtonFor(group){{const id=group?.dataset.extraId; return id?document.querySelector(`[data-extra-target="${{CSS.escape(id)}}"]`):null;}}
function ensureContextRowsPrepared(group){{if(!group||group.dataset.contextPrepared==='true')return; contextRows(group).forEach(row=>row.classList.add('hidden')); group.dataset.contextPrepared='true';}}
function updateContextButton(group){{const button=contextButtonFor(group); if(!button)return; const rows=contextRows(group), hidden=hiddenContextRows(group).length; if(rows.length&&hidden===0&&!group.classList.contains('hidden'))button.textContent=ui.hide_extra; else button.textContent=`${{button.dataset.extraLabel||button.dataset.originalText}} (${{hidden||rows.length}})`;}}
function revealContextRows(group,count=Infinity){{ensureContextRowsPrepared(group); if(!group)return 0; group.classList.remove('hidden'); const hidden=hiddenContextRows(group); const selected=group.dataset.contextSide==='before'?hidden.slice(Math.max(0,hidden.length-count)):hidden.slice(0,count); selected.forEach(row=>row.classList.remove('hidden')); updateContextButton(group); return selected.length;}}
function hideContextRows(group){{ensureContextRowsPrepared(group); if(!group)return; contextRows(group).forEach(row=>row.classList.add('hidden')); group.classList.add('hidden'); updateContextButton(group);}}
function bindContextButtons(){{document.querySelectorAll('[data-extra-target]').forEach(b=>{{b.dataset.originalText=b.textContent; b.addEventListener('click',()=>{{const group=document.querySelector(`.${{CSS.escape(b.dataset.extraTarget)}}`); if(!group)return; ensureContextRowsPrepared(group); if(hiddenContextRows(group).length)revealContextRows(group); else hideContextRows(group); renderReviewComments(); positionReviewComposer();}});}}); document.querySelectorAll('.extra-context').forEach(group=>{{ensureContextRowsPrepared(group); updateContextButton(group);}});}}
function visibleHunkCodeHeight(hunk){{const code=hunk.querySelector('.hunk-code'); return code?Array.from(code.children).reduce((sum,child)=>sum+child.getBoundingClientRect().height,0):0;}}
function hunkCommentContentHeight(hunk){{const comment=hunk.querySelector('.hunk-comment'); if(!comment)return 0; const style=getComputedStyle(comment); const content=comment.querySelector('.sticky')||comment; return content.getBoundingClientRect().height+(parseFloat(style.paddingTop)||0)+(parseFloat(style.paddingBottom)||0);}}
function estimatedDiffRowHeight(hunk){{const table=hunk.querySelector('table.diff'); if(!table)return 20; const visibleRows=table.querySelectorAll('tr:not(.hidden)').length; return visibleRows?Math.max(16,table.getBoundingClientRect().height/visibleRows):20;}}
function updateInlineReviewWidths(root=document){{root.querySelectorAll('.hunk-code').forEach(code=>{{const width=Math.max(280,Math.floor(code.clientWidth||0)); code.style.setProperty('--review-inline-width',`${{width}}px`);}});}}
function autoFitHunkContext(hunk){{if(window.matchMedia('(max-width:1100px)').matches)return; const groups=hunk.querySelectorAll('.extra-context'); if(!groups.length)return; groups.forEach(ensureContextRowsPrepared); const missing=hunkCommentContentHeight(hunk)-visibleHunkCodeHeight(hunk); if(missing<=8){{groups.forEach(updateContextButton); return;}} let rowsNeeded=Math.ceil(missing/estimatedDiffRowHeight(hunk)); const after=hunk.querySelector('.extra-context[data-context-side="after"]'); if(after&&rowsNeeded>0)rowsNeeded-=revealContextRows(after,Math.min(rowsNeeded,hiddenContextRows(after).length)); const before=hunk.querySelector('.extra-context[data-context-side="before"]'); if(before&&rowsNeeded>0)revealContextRows(before,Math.min(rowsNeeded,hiddenContextRows(before).length)); groups.forEach(updateContextButton);}}
function autoFitAllHunkContext(){{updateInlineReviewWidths(app); document.querySelectorAll('.hunk').forEach(autoFitHunkContext); updateInlineReviewWidths(app);}}
function rangeRef(file,start,end){{if(!file||!start||!end)return ''; const a=Math.min(start,end), b=Math.max(start,end); return `${{file}}:${{a}}:${{b}}`;}}
function currentRangeRef(){{return selection?rangeRef(selection.file,selection.start,selection.end):'';}}
function currentRefFor(file){{if(!selection||selection.file!==file)return ''; return currentRangeRef();}}
function hideReviewComposer(){{if(!reviewComposer)return; reviewComposer.classList.add('hidden'); if(composerRow){{document.body.appendChild(reviewComposer); composerRow.remove(); composerRow=null;}}}}
function selectedRows(){{if(!selection)return []; return Array.from(document.querySelectorAll(`tr.selectable.selected[data-file="${{CSS.escape(selection.file)}}"]`));}}
function updateComposerSaveState(){{const save=document.getElementById('commentSave'); if(save)save.disabled=!String(reviewComposerText?.value||'').trim();}}
function positionReviewComposer(){{if(!selection||activeTab!=='files'||!reviewComposer){{hideReviewComposer(); return;}} const rows=selectedRows(); if(!rows.length){{hideReviewComposer(); return;}} const anchor=rows[rows.length-1]; if(!anchor.offsetParent){{hideReviewComposer(); return;}} const hunkCode=anchor.closest('.hunk-code'); if(hunkCode)updateInlineReviewWidths(hunkCode.closest('.hunk')||hunkCode); if(reviewComposerTitle)reviewComposerTitle.textContent=`${{editingCommentId?ui.edit_comment_on:ui.add_comment_on}} ${{currentRangeRef()}}`; if(!composerRow){{composerRow=document.createElement('tr'); composerRow.className='review-composer-row'; composerRow.innerHTML='<td colspan="4"></td>';}} const cell=composerRow.firstElementChild; if(cell&&reviewComposer.parentNode!==cell)cell.appendChild(reviewComposer); if(anchor.nextSibling!==composerRow)anchor.parentNode?.insertBefore(composerRow,anchor.nextSibling); reviewComposer.classList.remove('hidden'); updateComposerSaveState();}}
function updateSelectionUi(){{document.querySelectorAll('tr.selected').forEach(r=>r.classList.remove('selected')); if(selection){{const a=Math.min(selection.start,selection.end), b=Math.max(selection.start,selection.end); document.querySelectorAll(`tr.selectable[data-file="${{CSS.escape(selection.file)}}"]`).forEach(r=>{{const line=Number(r.dataset.line); if(line>=a&&line<=b)r.classList.add('selected');}});}} positionReviewComposer();}}
function beginDraftSelection(file,line){{if(!selection||selection.file!==file){{selection={{file,start:line,end:line}}; editingCommentId=null; if(reviewComposerText)reviewComposerText.value='';}} else {{selection={{file,start:selection.start,end:line}}; editingCommentId=null;}} updateSelectionUi(); reviewComposerText?.focus();}}
function saveCurrentComment(button){{const text=String(reviewComposerText?.value||'').trim(); if(!selection||!text)return; const f=fileByPath(selection.file); if(!f)return; const start=Math.min(selection.start,selection.end), end=Math.max(selection.start,selection.end); const now=new Date().toISOString(); commentsState.comments=commentsState.comments||[]; if(editingCommentId){{const existing=commentsState.comments.find(comment=>comment.id===editingCommentId); if(existing){{existing.text=text; existing.start=start; existing.end=end; existing.updatedAt=now; existing.fingerprint=f.reviewFingerprint;}}}} else {{commentsState.comments.push({{id:`comment-${{Date.now()}}-${{Math.random().toString(36).slice(2)}}`,file:selection.file,fingerprint:f.reviewFingerprint,start,end,text,createdAt:now,updatedAt:now}});}} saveCommentsState(); editingCommentId=null; selection=null; if(reviewComposerText)reviewComposerText.value=''; hideReviewComposer(); updateSelectionUi(); renderReviewComments(); updateCommentSummaryUi(); if(button){{const old=button.textContent; button.textContent=ui.saved_comment; setTimeout(()=>button.textContent=old,1200);}}}}
function cancelCurrentComment(){{editingCommentId=null; selection=null; if(reviewComposerText)reviewComposerText.value=''; hideReviewComposer(); updateSelectionUi();}}
function commentAnchorRow(comment){{const rows=Array.from(document.querySelectorAll(`tr.selectable[data-file="${{CSS.escape(comment.file)}}"]`)).filter(row=>Number(row.dataset.line)===Number(comment.end)); return rows.find(row=>row.offsetParent!==null)||rows[0]||null;}}
function markCommentedRows(comment){{const a=Math.min(comment.start,comment.end), b=Math.max(comment.start,comment.end); document.querySelectorAll(`tr.selectable[data-file="${{CSS.escape(comment.file)}}"]`).forEach(row=>{{const line=Number(row.dataset.line); if(line>=a&&line<=b)row.classList.add('commented');}});}}
function renderReviewComments(){{document.querySelectorAll('.review-comment-row').forEach(row=>row.remove()); document.querySelectorAll('tr.commented').forEach(row=>row.classList.remove('commented')); validReviewComments().forEach(comment=>{{markCommentedRows(comment); if(comment.id===editingCommentId)return; const anchor=commentAnchorRow(comment); if(!anchor)return; const tr=document.createElement('tr'); tr.className='review-comment-row'; tr.dataset.commentId=comment.id; tr.innerHTML=`<td class="review-comment-cell" colspan="4"><div class="review-comment-card"><div class="review-comment-meta"><span class="review-comment-ref">${{esc(rangeRef(comment.file,comment.start,comment.end))}}</span><span class="review-comment-actions"><button data-edit-comment="${{esc(comment.id)}}">${{esc(ui.edit)}}</button><button data-delete-comment="${{esc(comment.id)}}">${{esc(ui.delete)}}</button></span></div><div class="review-comment-body">${{esc(comment.text)}}</div></div></td>`; anchor.parentNode?.insertBefore(tr,anchor.nextSibling);}}); document.querySelectorAll('[data-edit-comment]').forEach(button=>button.addEventListener('click',()=>editReviewComment(button.dataset.editComment))); document.querySelectorAll('[data-delete-comment]').forEach(button=>button.addEventListener('click',()=>deleteReviewComment(button.dataset.deleteComment))); updateCommentSummaryUi();}}
function editReviewComment(id){{const comment=validReviewComments().find(item=>item.id===id); if(!comment)return; editingCommentId=id; selection={{file:comment.file,start:comment.start,end:comment.end}}; if(reviewComposerText)reviewComposerText.value=comment.text; renderReviewComments(); updateSelectionUi(); reviewComposerText?.focus();}}
function deleteReviewComment(id){{commentsState.comments=(commentsState.comments||[]).filter(comment=>comment.id!==id); if(editingCommentId===id)cancelCurrentComment(); saveCommentsState(); renderReviewComments(); updateSelectionUi();}}
function formatReviewComments(){{return validReviewComments().map((comment,index)=>`${{index+1}}. ${{rangeRef(comment.file,comment.start,comment.end)}}\\n${{comment.text}}`).join('\\n\\n');}}
function updateCommentSummaryUi(){{const count=validReviewComments().length; if(copyAllCommentsButton){{copyAllCommentsButton.textContent=`${{ui.copy_all_comments}}${{count?` (${{count}})`:''}}`; copyAllCommentsButton.disabled=count===0;}} if(clearAllCommentsButton)clearAllCommentsButton.disabled=count===0;}}
function copyAllReviewComments(button){{const text=formatReviewComments(); if(!text){{if(button){{const old=button.textContent; button.textContent=ui.no_comments_to_copy; setTimeout(()=>button.textContent=old,1200);}} return;}} copyText(text,button,ui.comments_copied);}}
function clearAllReviewComments(button){{if(!validReviewComments().length)return; commentsState={{comments:[]}}; saveCommentsState(); editingCommentId=null; selection=null; if(reviewComposerText)reviewComposerText.value=''; hideReviewComposer(); updateSelectionUi(); renderReviewComments(); updateCommentSummaryUi(); if(button){{const old=button.textContent; button.textContent=ui.comments_cleared; setTimeout(()=>button.textContent=old,1200);}}}}
function bindSelection(){{document.querySelectorAll('tr.selectable').forEach(row=>row.addEventListener('click',()=>{{const file=row.dataset.file; const line=Number(row.dataset.line); beginDraftSelection(file,line);}})); document.getElementById('commentCancel')?.addEventListener('click',cancelCurrentComment); document.getElementById('commentSave')?.addEventListener('click',event=>saveCurrentComment(event.currentTarget)); reviewComposerText?.addEventListener('input',updateComposerSaveState); app.addEventListener('scroll',positionReviewComposer,{{passive:true}}); updateSelectionUi();}}
function copyText(text,button,successText=ui.copied){{if(!text)return; const done=()=>{{if(!button)return; const old=button.textContent; button.textContent=successText; setTimeout(()=>button.textContent=old,1200);}}; if(navigator.clipboard?.writeText)navigator.clipboard.writeText(text).then(done).catch(()=>fallbackCopy(text,done)); else fallbackCopy(text,done);}}
function fallbackCopy(text,done){{const ta=document.createElement('textarea'); ta.value=text; ta.style.position='fixed'; ta.style.opacity='0'; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); done();}}
function svgEl(name){{return document.createElementNS('http://www.w3.org/2000/svg',name);}}
function flowNodeTitle(node){{return node?.querySelector('.flow-node-title')?.textContent?.trim()||node?.dataset.nodeId||'';}}
function edgeDetails(edge,from,to){{const parts=[]; const route=`${{flowNodeTitle(from)}} -> ${{flowNodeTitle(to)}}`; if(route.trim()!=='->')parts.push(route); const label=localize(edge.label); if(label)parts.push(label); const when=localize(edge.when); if(when)parts.push(`${{ui.when}}: ${{when}}`); const evidence=(edge.evidence||[]).map(evidenceLabel).filter(Boolean); if(evidence.length)parts.push(`${{ui.evidence}}: ${{evidence.join(' | ')}}`); return parts.join('\\n');}}
function updateEdgeDetailPanel(button,label,details){{const section=button.closest('.arch-section'); const panel=section?.querySelector('.flow-edge-detail'); if(!panel)return; if(button.classList.contains('expanded')){{panel.classList.add('active'); panel.innerHTML=`<strong>${{esc(label)}}</strong><pre>${{esc(details)}}</pre>`;}} else if(!section.querySelector('.flow-edge-label.expanded')){{panel.classList.remove('active'); panel.textContent='';}}}}
function cubicPoint(path,t){{const mt=1-t; return {{x:mt*mt*mt*path.startX+3*mt*mt*t*path.c1x+3*mt*t*t*path.c2x+t*t*t*path.endX,y:mt*mt*mt*path.startY+3*mt*mt*t*path.c1y+3*mt*t*t*path.c2y+t*t*t*path.endY}};}}
function cubicDerivative(path,t){{const mt=1-t; return {{x:3*mt*mt*(path.c1x-path.startX)+6*mt*t*(path.c2x-path.c1x)+3*t*t*(path.endX-path.c2x),y:3*mt*mt*(path.c1y-path.startY)+6*mt*t*(path.c2y-path.c1y)+3*t*t*(path.endY-path.c2y)}};}}
function flowNodeRects(grid){{return Array.from(grid.querySelectorAll('.flow-node')).map(node=>({{left:node.offsetLeft-6,top:node.offsetTop-6,right:node.offsetLeft+node.offsetWidth+6,bottom:node.offsetTop+node.offsetHeight+6}}));}}
function labelRectAt(button,x,y){{const width=button.offsetWidth||90, height=button.offsetHeight||22; return {{left:x-width/2,top:y-height/2,right:x+width/2,bottom:y+height/2,width,height}};}}
function inflateRect(rect,padX,padY){{return {{left:rect.left-padX,top:rect.top-padY,right:rect.right+padX,bottom:rect.bottom+padY,width:rect.width+padX*2,height:rect.height+padY*2}};}}
function placedLabelRects(grid,current){{return Array.from(grid.querySelectorAll('.flow-edge-label')).filter(label=>label!==current).map(label=>inflateRect(labelRectAt(label,label.offsetLeft,label.offsetTop),8,6));}}
function overlapArea(a,b){{const width=Math.min(a.right,b.right)-Math.max(a.left,b.left), height=Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top); return width>0&&height>0?width*height:0;}}
function edgeLabelScore(candidate,box,nodeRects,labelRects,width,height){{let score=candidate.distance*18+Math.max(0,candidate.distance-64)**2*1.6; nodeRects.forEach(rect=>{{const overlap=overlapArea(box,rect); if(overlap>0)score+=200000+overlap*20;}}); labelRects.forEach(rect=>{{const overlap=overlapArea(box,rect); if(overlap>0)score+=300000+overlap*30;}}); if(box.left<0)score+=Math.abs(box.left)*1000; if(box.top<0)score+=Math.abs(box.top)*1000; if(box.right>width)score+=(box.right-width)*1000; if(box.bottom>height)score+=(box.bottom-height)*1000; return score;}}
function edgeLabelCandidates(button,from,to,path){{const candidates=[]; const seen=new Set(); const push=(x,y,anchorX,anchorY)=>{{const key=`${{Math.round(x)}}:${{Math.round(y)}}`; if(!seen.has(key)){{seen.add(key); candidates.push({{x,y,anchorX,anchorY,distance:Math.hypot(x-anchorX,y-anchorY)}});}}}}; const labelHeight=button.offsetHeight||22; [0.5,0.42,0.58,0.32,0.68,0.22,0.78].forEach(t=>{{const point=cubicPoint(path,t), derivative=cubicDerivative(path,t); const length=Math.hypot(derivative.x,derivative.y)||1; const nx=-derivative.y/length, ny=derivative.x/length; [18,-18,30,-30,46,-46,64,-64,86,-86,112,-112].forEach(offset=>push(point.x+nx*offset,point.y+ny*offset,point.x,point.y));}}); const sameRow=Math.abs(from.offsetTop-to.offsetTop)<8; const anchor=cubicPoint(path,0.5), top=Math.min(from.offsetTop,to.offsetTop), bottom=Math.max(from.offsetTop+from.offsetHeight,to.offsetTop+to.offsetHeight); if(sameRow){{[12,28,46,68].forEach(gap=>{{push(anchor.x,bottom+labelHeight/2+gap,anchor.x,anchor.y); push(anchor.x,top-labelHeight/2-gap,anchor.x,anchor.y);}});}} else {{[16,34,56,82].forEach(gap=>{{push(anchor.x,top-labelHeight/2-gap,anchor.x,anchor.y); push(anchor.x,bottom+labelHeight/2+gap,anchor.x,anchor.y);}});}} return candidates;}}
function placeEdgeLabel(button,grid,from,to,path){{const nodeRects=flowNodeRects(grid), labelRects=placedLabelRects(grid,button), width=Math.max(grid.scrollWidth,grid.clientWidth), height=Math.max(grid.scrollHeight,grid.clientHeight); let best=null; edgeLabelCandidates(button,from,to,path).forEach(candidate=>{{const box=labelRectAt(button,candidate.x,candidate.y), score=edgeLabelScore(candidate,box,nodeRects,labelRects,width,height); if(!best||score<best.score)best={{...candidate,score}};}}); if(best){{button.style.left=`${{best.x}}px`; button.style.top=`${{best.y}}px`; button.dataset.anchorX=String(best.anchorX); button.dataset.anchorY=String(best.anchorY); button.dataset.labelDistance=String(Math.round(best.distance)); button.dataset.collisionScore=String(best.score);}} return best;}}
function drawEdgeLeader(svg,button,placement){{if(!svg||!placement)return; const distance=placement.distance||0; const dot=svgEl('circle'); dot.setAttribute('class','flow-edge-anchor'); dot.setAttribute('cx',placement.anchorX); dot.setAttribute('cy',placement.anchorY); dot.setAttribute('r','2.4'); svg.appendChild(dot); if(distance<24)return; const leader=svgEl('path'); leader.setAttribute('class','flow-edge-leader'); leader.setAttribute('d',`M ${{placement.anchorX}} ${{placement.anchorY}} L ${{placement.x}} ${{placement.y}}`); svg.appendChild(leader);}}
function addEdgeLabel(layer,edge,from,to,pathSpec){{const label=localize(edge.label)||localize(edge.when); if(!label||!layer)return; const button=document.createElement('button'); button.type='button'; button.className='flow-edge-label'; button.textContent=label; const details=edgeDetails(edge,from,to); button.title=details; button.setAttribute('aria-label',details||label); button.style.left=`${{(pathSpec.startX+pathSpec.endX)/2}}px`; button.style.top=`${{(pathSpec.startY+pathSpec.endY)/2}}px`; button.addEventListener('click',event=>{{event.stopPropagation(); document.querySelectorAll('.flow-edge-label.expanded').forEach(el=>{{if(el!==button){{el.classList.remove('expanded'); updateEdgeDetailPanel(el,el.textContent,el.title);}}}}); button.classList.toggle('expanded'); updateEdgeDetailPanel(button,label,details);}}); layer.appendChild(button); const grid=layer.closest('.flow-grid'); const placement=placeEdgeLabel(button,grid,from,to,pathSpec); drawEdgeLeader(grid?.querySelector('.flow-svg'),button,placement);}}
function drawArchitectureEdges(){{document.querySelectorAll('.flow-grid').forEach(grid=>{{applyFlowZoom(grid,flowZoomFor(grid)); const svg=grid.querySelector('.flow-svg'); if(!svg)return; while(svg.firstChild)svg.removeChild(svg.firstChild); const labelLayer=grid.querySelector('.flow-edge-label-layer'); if(labelLayer)labelLayer.replaceChildren(); const width=Math.max(grid.scrollWidth,grid.clientWidth), height=Math.max(grid.scrollHeight,grid.clientHeight); svg.setAttribute('viewBox',`0 0 ${{width}} ${{height}}`); svg.setAttribute('width',width); svg.setAttribute('height',height); const markerId=`${{grid.dataset.flowId}}-arrow`; const defs=svgEl('defs'); const marker=svgEl('marker'); marker.setAttribute('id',markerId); marker.setAttribute('viewBox','0 0 10 10'); marker.setAttribute('refX','9'); marker.setAttribute('refY','5'); marker.setAttribute('markerWidth','6'); marker.setAttribute('markerHeight','6'); marker.setAttribute('orient','auto-start-reverse'); const markerPath=svgEl('path'); markerPath.setAttribute('d','M 0 0 L 10 5 L 0 10 z'); markerPath.setAttribute('fill','#64748b'); marker.appendChild(markerPath); defs.appendChild(marker); svg.appendChild(defs); const edges=decodeFiles(grid.dataset.edges); edges.forEach(edge=>{{const from=grid.querySelector(`[data-node-id="${{CSS.escape(edge.from)}}"]`); const to=grid.querySelector(`[data-node-id="${{CSS.escape(edge.to)}}"]`); if(!from||!to)return; const startX=from.offsetLeft+from.offsetWidth, startY=from.offsetTop+from.offsetHeight/2, endX=to.offsetLeft, endY=to.offsetTop+to.offsetHeight/2; const dx=Math.max(36,Math.abs(endX-startX)/2); const pathSpec={{startX,startY,c1x:startX+dx,c1y:startY,c2x:endX-dx,c2y:endY,endX,endY}}; const path=svgEl('path'); path.setAttribute('d',`M ${{startX}} ${{startY}} C ${{pathSpec.c1x}} ${{pathSpec.c1y}}, ${{pathSpec.c2x}} ${{pathSpec.c2y}}, ${{endX}} ${{endY}}`); path.setAttribute('fill','none'); path.setAttribute('stroke','#64748b'); path.setAttribute('stroke-width','1.5'); path.setAttribute('marker-end',`url(#${{markerId}})`); svg.appendChild(path); addEdgeLabel(labelLayer,edge,from,to,pathSpec);}});}});}}
function bindArchitecture(){{document.querySelectorAll('[data-arch-files]').forEach(el=>el.addEventListener('click',()=>focusArchitectureFiles(decodeFiles(el.dataset.archFiles)))); document.querySelectorAll('.arch-clear-focus').forEach(b=>b.addEventListener('click',clearArchitectureFocus)); initializeFlowZooms(); bindFlowZoomControls(); updateArchitectureFocusUi(); if(activeTab==='architecture')drawArchitectureEdges();}}
function updateReviewUi(){{const count=reviewedCount(); const total=data.files.length; const pending=Math.max(0,total-count); const progress=document.getElementById('reviewProgress'); if(progress){{const pct=total?Math.round((count/total)*100):0; progress.textContent=`${{count}} / ${{total}} ${{ui.reviewed_progress_suffix}}`; progress.style.setProperty('--progress',`${{pct}}%`);}} if(reviewQueueSummary)reviewQueueSummary.textContent=`${{count}} / ${{total}} ${{ui.reviewed_progress_suffix}}`; if(reviewedCountLabel)reviewedCountLabel.textContent=String(count); if(pendingCountLabel)pendingCountLabel.textContent=String(pending); document.querySelectorAll('[data-reviewed-status]').forEach(el=>{{const reviewed=isReviewed(el.dataset.reviewedStatus); el.textContent=reviewed?ui.reviewed:ui.pending; el.classList.toggle('reviewed',reviewed);}}); document.querySelectorAll('[data-reviewed-toggle]').forEach(b=>{{const reviewed=isReviewed(b.dataset.reviewedToggle); b.textContent=reviewed?ui.mark_pending:ui.mark_reviewed; b.classList.toggle('reviewed',reviewed);}}); document.querySelectorAll('section.file[data-file-path]').forEach(s=>s.classList.toggle('reviewed',isReviewed(s.dataset.filePath))); document.querySelectorAll('.navfile[data-file-path]').forEach(a=>{{const reviewed=isReviewed(a.dataset.filePath); a.classList.toggle('reviewed',reviewed); const mark=a.querySelector('.review-mark'); if(mark)mark.textContent=reviewed?'[x]':'[ ]'; const status=a.querySelector('.nav-status'); if(status)status.textContent=reviewed?ui.reviewed:ui.pending;}}); applyFilters();}}
function updateFileToggleButton(button, collapsed){{if(!button)return; button.setAttribute('aria-expanded',String(!collapsed)); button.title=collapsed?ui.expand:ui.collapse; button.setAttribute('aria-label',collapsed?ui.expand:ui.collapse);}}
function updateFileHeaderToggle(header, collapsed){{if(!header)return; header.title=collapsed?ui.expand:ui.collapse;}}
function scrollClosedFileHeaderIntoView(file){{if(!file)return; requestAnimationFrame(()=>{{const container=scrollContainerForTab('files'); if(container&&container.scrollTo&&container.contains(file)){{const fileBox=file.getBoundingClientRect(); const containerBox=container.getBoundingClientRect(); const visualOffset=Math.min(72,Math.max(36,container.clientHeight*.08)); const target=container.scrollTop+fileBox.top-containerBox.top-visualOffset; container.scrollTo({{top:Math.max(0,target),behavior:'smooth'}});}} else {{file.scrollIntoView({{block:'start',behavior:'smooth'}});}}}});}}
function setFileBodyExpanded(body, expanded, persist=false, scrollOnCollapse=false){{if(!body)return; const file=body.closest('.file'); body.classList.toggle('hidden',!expanded); file?.classList.toggle('collapsed',!expanded); updateFileToggleButton(document.querySelector(`[data-target="${{body.id}}"]`),!expanded); updateFileHeaderToggle(document.querySelector(`[data-file-header-toggle="${{body.id}}"]`),!expanded); if(!expanded&&selection?.file===file?.dataset.filePath){{selection=null; updateSelectionUi();}} if(expanded)requestAnimationFrame(()=>{{body.querySelectorAll('.hunk').forEach(autoFitHunkContext); renderReviewComments();}}); else {{requestAnimationFrame(renderReviewComments); if(scrollOnCollapse)scrollClosedFileHeaderIntoView(file);}} requestAnimationFrame(positionReviewComposer); if(persist&&file?.dataset.filePath)setFileExpanded(file.dataset.filePath,expanded);}}
function restoreFileExpansionState(){{document.querySelectorAll('.file-body').forEach(body=>{{const path=body.closest('.file')?.dataset.filePath; setFileBodyExpanded(body,Boolean(path&&isFileExpanded(path)),false);}});}}
function isHeaderInteractiveTarget(target){{return Boolean(target.closest('button,a,input,textarea,select,label,[role="button"]'));}}
function bindFileHeaderToggles(){{document.querySelectorAll('[data-file-header-toggle]').forEach(header=>header.addEventListener('click',event=>{{if(isHeaderInteractiveTarget(event.target))return; const body=document.getElementById(header.dataset.fileHeaderToggle); if(!body)return; setFileBodyExpanded(body,body.classList.contains('hidden'),true,true);}}));}}
function bindReviewState(){{document.querySelectorAll('[data-reviewed-toggle]').forEach(b=>b.addEventListener('click',()=>setReviewed(b.dataset.reviewedToggle,!isReviewed(b.dataset.reviewedToggle)))); document.getElementById('pendingOnly').addEventListener('change',e=>{{showPendingOnly=e.target.checked; applyFilters();}}); document.getElementById('clearReviewed').addEventListener('click',clearReviewed); copyAllCommentsButton?.addEventListener('click',event=>copyAllReviewComments(event.currentTarget)); clearAllCommentsButton?.addEventListener('click',event=>clearAllReviewComments(event.currentTarget)); updateCommentSummaryUi();}}
function render(){{updateStickyOffsets(); summaryPanel.innerHTML=overview(); architecturePanel.innerHTML=architecture(); app.innerHTML=data.files.map(file).join(''); navFiles.innerHTML=data.files.map((f,i)=>`<a class="navfile" href="#file-${{i}}" title="${{esc(f.new_path)}}" data-file-path="${{esc(f.new_path)}}" data-search="${{esc((f.new_path+' '+f.category[data.language]+' '+f.note[data.language]).toLowerCase())}}"><span class="review-mark">[ ]</span><span class="navfile-main"><span class="navfile-path">${{esc(shortPath(f.new_path))}}</span><small>${{esc(f.category[data.language])}} · ${{esc(f.status)}}</small></span><span class="nav-delta"><span class="adds">+${{esc(f.numstat.additions)}}</span> <span class="dels">-${{esc(f.numstat.deletions)}}</span></span><span class="nav-status">${{esc(ui.pending)}}</span></a>`).join(''); restoreFileExpansionState(); document.querySelectorAll('.toggle').forEach(b=>{{const el=document.getElementById(b.dataset.target); updateFileToggleButton(b,el?.classList.contains('hidden')??true); b.addEventListener('click',()=>{{const target=document.getElementById(b.dataset.target); if(!target)return; setFileBodyExpanded(target,target.classList.contains('hidden'),true,true);}});}}); bindFileHeaderToggles(); bindTabs(); bindFileNavigation(); bindArchitecture(); bindContextButtons(); bindSelection(); bindReviewState(); updateReviewUi(); renderReviewComments(); applySyntaxHighlighting(app); requestAnimationFrame(()=>{{if(activeTab==='files'){{autoFitAllHunkContext(); renderReviewComments();}} if(activeTab==='architecture')drawArchitectureEdges();}});}}
function applyFilters(){{const q=document.getElementById('filter').value.trim().toLowerCase(); document.querySelectorAll('section.file').forEach(s=>{{const hideByQuery=Boolean(q&&!s.dataset.search.includes(q)); const hideByReview=Boolean(showPendingOnly&&isReviewed(s.dataset.filePath)); const hideByArchitecture=Boolean(architectureFocus&&!architectureFocus.has(s.dataset.filePath)); s.classList.toggle('hidden',hideByQuery||hideByReview||hideByArchitecture);}}); document.querySelectorAll('.navfile').forEach(a=>{{const path=a.dataset.filePath; const hideByQuery=Boolean(q&&!a.dataset.search.includes(q)); const hideByReview=Boolean(path&&showPendingOnly&&isReviewed(path)); const hideByArchitecture=Boolean(path&&architectureFocus&&!architectureFocus.has(path)); a.classList.toggle('hidden',hideByQuery||hideByReview||hideByArchitecture);}});}}
function filter(q){{applyFilters();}}
document.getElementById('filter').addEventListener('input',e=>filter(e.target.value)); document.getElementById('expandAll').addEventListener('click',()=>document.querySelectorAll('.file-body').forEach(b=>setFileBodyExpanded(b,true,true))); document.getElementById('collapseAll').addEventListener('click',()=>document.querySelectorAll('.file-body').forEach(b=>setFileBodyExpanded(b,false,true))); sidebarToggle?.addEventListener('click',()=>{{setSidebarCollapsed(!sidebarCollapsed); saveSidebarState();}}); window.addEventListener('resize',()=>{{updateStickyOffsets(); if(activeTab==='architecture')drawArchitectureEdges(); if(activeTab==='files'){{autoFitAllHunkContext(); positionReviewComposer();}}}}); window.addEventListener('load',()=>{{applySyntaxHighlighting(app); if(activeTab==='architecture')drawArchitectureEdges(); if(activeTab==='files'){{autoFitAllHunkContext(); renderReviewComments(); positionReviewComposer();}}}}); setSidebarCollapsed(sidebarCollapsed); render();
</script>
</body>
</html>"""


def validate_html(path: Path, expected_files: int) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="walkthrough-data" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise RuntimeError(f"{path} does not contain embedded walkthrough JSON")
    data = json.loads(match.group(1))
    if len(data.get("files", [])) != expected_files:
        raise RuntimeError(f"{path} has {len(data.get('files', []))} files, expected {expected_files}")
    if "reviewStateKey" not in data:
        raise RuntimeError(f"{path} does not include a review state key")
    if any("reviewFingerprint" not in file for file in data.get("files", [])):
        raise RuntimeError(f"{path} has files without review fingerprints")
    lowered = text.lower()
    if "highlight.min.js" not in lowered or "applysyntaxhighlighting" not in lowered:
        raise RuntimeError(f"{path} does not include syntax highlighting support")
    if "data-tab-panel" not in lowered or "setactivetab" not in lowered or "tabscrollpositions" not in lowered:
        raise RuntimeError(f"{path} does not include independent walkthrough tabs")
    if "localstorage" not in lowered or "data-reviewed-toggle" not in lowered:
        raise RuntimeError(f"{path} does not include persistent review controls")
    if "review-composer" not in lowered or "renderreviewcomments" not in lowered or "copyallcomments" not in lowered:
        raise RuntimeError(f"{path} does not include persistent review comment controls")
    if "file-sticky" not in lowered or "position:sticky" not in lowered:
        raise RuntimeError(f"{path} does not include sticky file headers")
    architecture = data.get("architecture") or {}
    if architecture.get("sections") or architecture.get("glossary"):
        if "flow-grid" not in lowered or "drawarchitectureedges" not in lowered or "data-arch-files" not in lowered:
            raise RuntimeError(f"{path} does not include architecture flow controls")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getcwd(), help="Repository checkout. Defaults to current directory.")
    parser.add_argument("--pr", help="GitHub PR number or URL. Uses gh for metadata.")
    parser.add_argument("--base", default="master", help="Base ref for git diff. Default: master.")
    parser.add_argument("--head", default="HEAD", help="Head ref for git diff. Default: HEAD.")
    parser.add_argument("--range", help="Exact git diff range. Overrides --base/--head for diffing.")
    parser.add_argument("--context", action="append", help="Plan/design/context file to embed. Can be repeated.")
    parser.add_argument("--notes-json", help="Optional custom notes JSON with files, regex patterns, and architecture flows.")
    parser.add_argument("--out", help="Output directory. Defaults to a temporary folder outside the repo.")
    parser.add_argument("--title", help="Override artifact title.")
    parser.add_argument("--surrounding-lines", type=int, default=12, help="Extra unchanged lines available behind each hunk's expand buttons. Default: 12.")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a git checkout: {repo}")

    model, outdir = build_model(repo, args)
    expected_files = len(model["files"])
    (outdir / "index.html").write_text(render_html(model, "en"), encoding="utf-8")
    (outdir / "index.es.html").write_text(render_html(model, "es"), encoding="utf-8")
    manifest = {
        "generated_at": model["generatedAt"],
        "repo": str(repo),
        "range": model["range"],
        "head": model["headSha"],
        "file_count": expected_files,
        "files": {
            "html_en": str(outdir / "index.html"),
            "html_es": str(outdir / "index.es.html"),
            "raw_diff": str(outdir / "pr.diff"),
            "diff_stat": str(outdir / "diff-stat.txt"),
            "commits": str(outdir / "commits.txt"),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    validate_html(outdir / "index.html", expected_files)
    validate_html(outdir / "index.es.html", expected_files)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
