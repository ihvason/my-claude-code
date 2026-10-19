"""
Shared terminal rendering primitives, reused by commands / permissions / hooks
and depending only on rich. They live here to break the circular dependency
between commands and permissions/hooks, so callers can import them at the top of
a file.
"""
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text


# Shared Console instance; handles ANSI colors across platforms (so even old
# Windows terminals render correctly).
# highlight=False turns off Rich's built-in number/string highlighting so that
# only the markup we add by hand takes effect.
console = Console(highlight=False)


def print_step(label: str, content: str = "") -> None:
    """
    Print one intermediate step block: the label takes a whole line, the content
    is indented 2 columns with Padding (so automatically wrapped continuation
    lines stay indented too), and a blank line is left at the end.
    """
    console.print(label)
    if content:
        console.print(Padding(content, (0, 0, 0, 2)))
    console.print()


def print_welcome_banner(title: str) -> None:
    """
    Print the startup banner: the title inside a colored rich.Panel, together
    with a /help hint.
    """
    body = Text.assemble(
        ("✻ ", "bright_magenta"),
        (f"Welcome to {title}\n\n", "bold"),
        ("  /help ", "cyan"),
        ("list available commands", "dim"),
    )
    console.print(Panel(body, border_style="bright_blue", padding=(0, 1), width=48))
    console.print()


def print_system_text(text: str) -> None:
    """
    Echo a piece of system-injected text (reminder / job notification) in a uniform dim style.
    """
    print_step("[dim]◇ system[/]", f"[dim]{text[:200]}[/]")
