"""
Import ChatGPT/Claude conversations.json to markdown files.

This module converts conversation exports from ChatGPT or Claude into the
markdown format expected by auto-anki's conversation harvesting pipeline.

Usage:
    auto-anki-import ~/Downloads/conversations.json
    auto-anki-import ~/Downloads/conversations.json -v --run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ExportStats:
    """Statistics from an export run."""

    written: int = 0  # New files created
    updated: int = 0  # Existing files with changed content
    skipped: int = 0  # Existing files unchanged

    @property
    def total(self) -> int:
        return self.written + self.updated + self.skipped


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 120) -> str:
    """Create a filesystem-friendly slug from a title."""
    text = text.strip().lower()
    # Replace separators with dashes
    text = re.sub(r"[\s/\\:]+", "-", text)
    # Remove invalid filename chars
    text = re.sub(r"[^a-z0-9._-]", "", text)
    # Collapse multiple dashes
    text = re.sub(r"-+", "-", text)
    return text[:max_len] or "untitled"


def strip_citations(text: str) -> str:
    """Remove ChatGPT citation annotations from text.

    Citations use Unicode private use area characters:
    - U+E200: start citation
    - U+E201: end citation
    - U+E202: separator within citation
    The text between markers contains reference info like 'cite', 'turn0search2', etc.
    """
    # Remove citation blocks: everything from \ue200 to \ue201 inclusive
    text = re.sub(r"\ue200[^\ue201]*\ue201", "", text)
    return text


def convert_latex_delimiters(text: str) -> str:
    r"""Convert LaTeX delimiters to Obsidian-compatible format.

    - \( ... \) → $...$ (inline math, spaces trimmed)
    - \[ ... \] → $$...$$ (display math, spaces trimmed)

    Handles multiline expressions using re.DOTALL flag.
    """
    # \[ ... \] → $$ ... $$ (display math, trim spaces, allow newlines)
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # \( ... \) → $ ... $ (inline math, trim spaces, allow newlines)
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def fmt_unix_timestamp(ts: Optional[float]) -> str:
    """Format Unix timestamp to readable string."""
    if ts is None:
        return ""
    try:
        # Local time for readability
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def fmt_iso_timestamp(iso_str: Optional[str]) -> str:
    """Format ISO 8601 timestamp string to readable format."""
    if not iso_str:
        return ""
    try:
        # Parse ISO format and convert to local time
        dt_obj = dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso_str)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(conversations: List[Dict[str, Any]]) -> Literal["chatgpt", "claude"]:
    """Detect whether this is ChatGPT or Claude format.

    ChatGPT: has 'mapping', 'title' keys
    Claude: has 'chat_messages', 'uuid' keys
    """
    if not conversations:
        return "chatgpt"  # default

    first_conv = conversations[0]
    # Claude has 'chat_messages' and 'uuid', ChatGPT has 'mapping' and 'title'
    if "chat_messages" in first_conv and "uuid" in first_conv:
        return "claude"
    return "chatgpt"


# ---------------------------------------------------------------------------
# ChatGPT rendering
# ---------------------------------------------------------------------------


def extract_text_and_assets(content: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (text, asset_pointers) from a message content object."""
    ctype = content.get("content_type")
    if ctype == "text":
        parts = content.get("parts") or []
        # parts is a list of strings
        text = "\n\n".join(p for p in parts if isinstance(p, str))
        text = strip_citations(text)
        text = convert_latex_delimiters(text)
        return text, []
    if ctype == "multimodal_text":
        parts = content.get("parts") or []
        texts: List[str] = []
        assets: List[str] = []
        for p in parts:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict):
                if p.get("content_type") == "image_asset":
                    ap = p.get("asset_pointer")
                    if isinstance(ap, str):
                        assets.append(ap)
        text = "\n\n".join(texts)
        text = strip_citations(text)
        text = convert_latex_delimiters(text)
        return text, assets
    # Fallback: try to stringify
    return json.dumps(content, ensure_ascii=False), []


def is_visually_hidden(msg: Dict[str, Any]) -> bool:
    """Check if a message is hidden from the conversation UI."""
    meta = msg.get("metadata") or {}
    return bool(meta.get("is_visually_hidden_from_conversation"))


def iter_visible_messages(mapping: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield visible (non-hidden, non-empty) messages from conversation mapping."""
    for node in mapping.values():
        msg = node.get("message")
        if not msg:
            continue
        if is_visually_hidden(msg):
            continue
        # Skip empty placeholder messages with no text content
        content = msg.get("content") or {}
        text, assets = extract_text_and_assets(content)
        if not text.strip() and not assets:
            continue
        yield msg


def render_chatgpt_conversation(
    conv: Dict[str, Any], base_url: str = "https://chatgpt.com"
) -> str:
    """Render ChatGPT conversation to markdown format."""
    title = conv.get("title") or "Untitled Chat"
    conv_id = conv.get("id") or conv.get("conversation_id")
    create = fmt_unix_timestamp(conv.get("create_time"))
    update = fmt_unix_timestamp(conv.get("update_time"))
    mapping = conv.get("mapping") or {}

    # Collect and sort visible messages
    msgs = list(iter_visible_messages(mapping))
    msgs.sort(key=lambda m: (m.get("create_time") or 0.0, m.get("id") or ""))

    lines: List[str] = []
    lines.append(f"# {title}")
    meta_bits = []
    # Conversation URL back to ChatGPT UI, when possible
    if conv_id and base_url:
        url = f"{base_url.rstrip('/')}/c/{conv_id}"
        meta_bits.append(f"URL: {url}")
    if create:
        meta_bits.append(f"Created: {create}")
    if update:
        meta_bits.append(f"Updated: {update}")
    if meta_bits:
        lines.append("")
        for bit in meta_bits:
            lines.append(f"- {bit}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for m in msgs:
        role = ((m.get("author") or {}).get("role") or "").strip() or "unknown"
        ts = fmt_unix_timestamp(m.get("create_time"))
        header = f"[{ts}] {role}:" if ts else f"{role}:"
        lines.append(header)
        content = m.get("content") or {}
        text, assets = extract_text_and_assets(content)
        if text.strip():
            lines.append(text.strip())
        for ap in assets:
            # Note the asset pointer so users can locate the image in the folder
            lines.append(f"\n[image asset: {ap}]\n")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Claude rendering
# ---------------------------------------------------------------------------


def render_claude_conversation(
    conv: Dict[str, Any], base_url: str = "https://claude.ai"
) -> str:
    """Render a Claude conversation to Markdown."""
    title = conv.get("name") or "Untitled Chat"
    conv_uuid = conv.get("uuid")
    created = fmt_iso_timestamp(conv.get("created_at"))
    updated = fmt_iso_timestamp(conv.get("updated_at"))

    lines: List[str] = []
    lines.append(f"# {title}")
    meta_bits = []

    # Conversation URL back to Claude UI
    if conv_uuid and base_url:
        url = f"{base_url.rstrip('/')}/chat/{conv_uuid}"
        meta_bits.append(f"URL: {url}")
    if created:
        meta_bits.append(f"Created: {created}")
    if updated:
        meta_bits.append(f"Updated: {updated}")

    if meta_bits:
        lines.append("")
        for bit in meta_bits:
            lines.append(f"- {bit}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Process messages
    chat_messages = conv.get("chat_messages") or []
    for msg in chat_messages:
        sender = msg.get("sender") or "unknown"
        created_at = fmt_iso_timestamp(msg.get("created_at"))
        header = f"[{created_at}] {sender}:" if created_at else f"{sender}:"
        lines.append(header)

        # Process content blocks
        content_blocks = msg.get("content") or []
        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                text = strip_citations(text)
                text = convert_latex_delimiters(text)
                if text.strip():
                    lines.append(text.strip())

            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                thinking = strip_citations(thinking)
                thinking = convert_latex_delimiters(thinking)
                if thinking.strip():
                    lines.append("<details>")
                    lines.append("<summary>Thinking</summary>")
                    lines.append("")
                    lines.append(thinking.strip())
                    lines.append("")
                    lines.append("</details>")

            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_msg = block.get("message", "")
                lines.append(f"**[Tool Use: {tool_name}]** {tool_msg}")

            elif block_type == "tool_result":
                tool_name = block.get("name", "unknown")
                # tool_result content is itself an array of content blocks
                result_content = block.get("content", [])
                if result_content:
                    lines.append(f"**[Tool Result: {tool_name}]**")
                    for result_block in result_content:
                        if (
                            isinstance(result_block, dict)
                            and result_block.get("type") == "text"
                        ):
                            result_text = result_block.get("text", "")
                            result_text = strip_citations(result_text)
                            result_text = convert_latex_delimiters(result_text)
                            if result_text.strip():
                                lines.append(result_text.strip())

        # Handle attachments
        attachments = msg.get("attachments") or []
        for att in attachments:
            fname = att.get("file_name", "unknown")
            ftype = att.get("file_type", "")
            fsize = att.get("file_size", 0)
            lines.append(f"\n**[Attachment: {fname}]** ({ftype}, {fsize} bytes)\n")
            # Optionally include extracted content
            extracted = att.get("extracted_content")
            if extracted:
                lines.append("<details>")
                lines.append("<summary>Attachment content</summary>")
                lines.append("")
                lines.append("```")
                lines.append(extracted[:1000])  # Limit to first 1000 chars
                if len(extracted) > 1000:
                    lines.append("... (truncated)")
                lines.append("```")
                lines.append("</details>")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_file_if_changed(
    path: Path, content: str
) -> Literal["written", "updated", "skipped"]:
    """Write file only if content has changed or doesn't exist.

    Returns status: 'written' (new), 'updated' (changed), or 'skipped' (unchanged).
    """
    if path.exists():
        try:
            existing_content = path.read_text(encoding="utf-8")
            if existing_content == content:
                return "skipped"
            # Content different, overwrite
            path.write_text(content, encoding="utf-8")
            return "updated"
        except Exception:
            # If read fails, just write
            path.write_text(content, encoding="utf-8")
            return "updated"
    else:
        # File doesn't exist, create new
        path.write_text(content, encoding="utf-8")
        return "written"


def set_file_times(path: Path, when: Optional[dt.datetime]) -> None:
    """Set file mtime/atime and, on macOS, the creation date.

    - Always sets atime/mtime via os.utime when a timestamp is provided.
    - On macOS, attempts to set the creation date using `SetFile -d` if available.
    """
    if when is None:
        return

    ts = when.timestamp()
    try:
        os.utime(path, (ts, ts))
    except Exception:
        pass

    if platform.system() == "Darwin":
        # Prefer discovered location, otherwise common Developer Tools paths
        setfile = shutil.which("SetFile")
        if not setfile:
            for candidate in (
                "/usr/bin/SetFile",
                "/Applications/Xcode.app/Contents/Developer/usr/bin/SetFile",
                "/Library/Developer/CommandLineTools/usr/bin/SetFile",
            ):
                if os.path.exists(candidate):
                    setfile = candidate
                    break

        if setfile:
            # SetFile expects local time in MM/DD/YYYY HH:MM:SS
            date_str = when.strftime("%m/%d/%Y %H:%M:%S")
            try:
                subprocess.run(
                    [setfile, "-d", date_str, str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                # Ignore failures silently; mtime/atime are still set.
                pass


# ---------------------------------------------------------------------------
# Config loading (shared with auto_anki_agent)
# ---------------------------------------------------------------------------


def _load_config() -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load optional configuration for default paths and settings.

    Search order:
    1. Path from AUTO_ANKI_CONFIG (if set)
    2. ./auto_anki_config.json in current working directory
    3. ~/.auto_anki_config.json in the user home directory

    Returns (config_dict, config_path) or ({}, None) if not found.
    """
    candidates: List[Path] = []
    env_path = os.getenv("AUTO_ANKI_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "auto_anki_config.json")
    candidates.append(Path.home() / ".auto_anki_config.json")

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
            return data, path
        except json.JSONDecodeError:
            print(f"Warning: Ignoring invalid config file (JSON parse error): {path}")
            break

    return {}, None


def _resolve_path(path_str: str, config_dir: Optional[Path]) -> Path:
    """Resolve a path, expanding ~ and relative paths."""
    p = Path(path_str).expanduser()
    if not p.is_absolute() and config_dir:
        p = config_dir / p
    return p.resolve()


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_conversations(
    json_path: Path,
    output_dir: Path,
    *,
    base_url: Optional[str] = None,
    workers: Optional[int] = None,
    verbose: bool = False,
) -> ExportStats:
    """
    Export conversations.json to markdown files.

    Creates nested directory structure: output_dir/YYYY/MM/DD/YYYY-MM-DD_slug.md

    Args:
        json_path: Path to conversations.json file
        output_dir: Root directory to write markdown files
        base_url: Base URL for conversation links (auto-detected if None)
        workers: Number of parallel worker threads (auto if None)
        verbose: Print progress during export

    Returns:
        ExportStats with counts of written/updated/skipped files
    """
    # Load and parse JSON
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        raise ValueError("Unexpected JSON format: expected a list of conversations")

    # Detect format
    conv_format = detect_format(conversations)
    if verbose:
        print(f"Detected format: {conv_format}")

    # Auto-set base_url if not specified
    if base_url is None:
        if conv_format == "claude":
            base_url = "https://claude.ai"
        else:
            base_url = "https://chatgpt.com"
        if verbose:
            print(f"Using base URL: {base_url}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thread-safe stats
    stats = ExportStats()
    stats_lock = threading.Lock()
    warned_setfile_missing = False
    warn_lock = threading.Lock()

    def conv_datetime(conv_obj: Dict[str, Any]) -> Optional[dt.datetime]:
        """Get conversation datetime, handling both formats."""
        if conv_format == "claude":
            iso_str = conv_obj.get("created_at") or conv_obj.get("updated_at")
            if iso_str is None:
                return None
            try:
                return dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            except Exception:
                return None
        else:
            ts = conv_obj.get("create_time") or conv_obj.get("update_time")
            if ts is None:
                return None
            try:
                return dt.datetime.fromtimestamp(ts)
            except Exception:
                return None

    def process_conversation(
        idx: int, conv: Dict[str, Any]
    ) -> Optional[Tuple[str, Path]]:
        nonlocal warned_setfile_missing

        if not isinstance(conv, dict):
            return None

        # Get title based on format
        if conv_format == "claude":
            title = conv.get("name") or f"chat-{idx:04d}"
        else:
            title = conv.get("title") or f"chat-{idx:04d}"

        slug = slugify(title) or f"chat-{idx:04d}"
        dt_obj = conv_datetime(conv)

        if dt_obj is None:
            # Undated bucket
            day_dir = output_dir / "undated"
            day_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{slug}.md"
        else:
            year = f"{dt_obj.year:04d}"
            month = f"{dt_obj.month:02d}"
            day = f"{dt_obj.day:02d}"
            day_dir = output_dir / year / month / day
            day_dir.mkdir(parents=True, exist_ok=True)
            date_prefix = dt_obj.strftime("%Y-%m-%d")
            fname = f"{date_prefix}_{slug}.md"

        # Render using appropriate function
        if conv_format == "claude":
            md = render_claude_conversation(conv, base_url=base_url)
        else:
            md = render_chatgpt_conversation(conv, base_url=base_url)

        out_path = day_dir / fname
        status = write_file_if_changed(out_path, md)

        # Set file timestamps only when writing or updating
        if status in ("written", "updated"):
            set_file_times(out_path, dt_obj)

        return (status, out_path)

    # Parallelize processing
    auto_workers = min(32, (os.cpu_count() or 4) + 4)
    env_workers = os.environ.get("EXPORT_MD_WORKERS")
    actual_workers = workers if workers and workers > 0 else (
        int(env_workers) if (env_workers and env_workers.isdigit()) else auto_workers
    )

    with ThreadPoolExecutor(max_workers=actual_workers) as ex:
        futures = [
            ex.submit(process_conversation, idx, conv)
            for idx, conv in enumerate(conversations, start=1)
        ]
        for fut in as_completed(futures):
            result = fut.result()  # surface exceptions
            if result:
                status, _ = result
                with stats_lock:
                    if status == "written":
                        stats.written += 1
                    elif status == "updated":
                        stats.updated += 1
                    else:
                        stats.skipped += 1

                    if verbose and stats.total % 50 == 0:
                        print(
                            f"Processed {stats.total} files "
                            f"(written: {stats.written}, updated: {stats.updated}, "
                            f"skipped: {stats.skipped})..."
                        )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for auto-anki-import."""
    parser = argparse.ArgumentParser(
        description="Import ChatGPT/Claude conversations.json to markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auto-anki-import ~/Downloads/conversations.json
  auto-anki-import ~/Downloads/conversations.json -v
  auto-anki-import ~/Downloads/conversations.json -o ~/chatgpt-archive
  auto-anki-import ~/Downloads/conversations.json --run
        """,
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to conversations.json file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: chat_root from config)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for conversation links (auto-detected if not specified)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker threads (default: auto)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress during export",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run auto-anki after import completes",
    )
    args = parser.parse_args(argv)

    # Validate input path
    json_path = args.json_path.expanduser().resolve()
    if not json_path.exists():
        print(f"Error: Input file not found: {json_path}", file=sys.stderr)
        return 1

    # Determine output directory
    if args.output:
        output_dir = args.output.expanduser().resolve()
    else:
        # Load from config
        config, config_path = _load_config()
        chat_root = config.get("chat_root")
        if chat_root:
            config_dir = config_path.parent if config_path else None
            output_dir = _resolve_path(chat_root, config_dir)
        else:
            # Default fallback
            output_dir = Path(
                "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt"
            ).expanduser()

    if args.verbose:
        print(f"Input: {json_path}")
        print(f"Output: {output_dir}")

    # Run export
    try:
        stats = export_conversations(
            json_path,
            output_dir,
            base_url=args.base_url,
            workers=args.workers,
            verbose=args.verbose,
        )
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(
        f"Done. Processed {stats.total} conversations: "
        f"{stats.written} new, {stats.updated} updated, {stats.skipped} unchanged."
    )
    print(f"Output directory: {output_dir}")

    # Optionally run auto-anki
    if args.run:
        print("\nRunning auto-anki...")
        # Import and run main
        try:
            from auto_anki.cli import main as auto_anki_main

            return auto_anki_main()
        except ImportError:
            # Fallback to subprocess
            import subprocess

            result = subprocess.run(["auto-anki", "--unprocessed-only"])
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
