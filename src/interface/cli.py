"""CLI interface for ReAgt using Rich."""

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel

from src.agent.coach import generate_plan, save_plan
from src.memory.profile import create_profile, save_profile

console = Console()

AVAILABLE_SPORTS = ["running", "cycling", "swimming", "gym"]


def onboard_athlete() -> dict:
    """Interactive CLI to collect athlete info and create a profile."""
    console.print(
        Panel("[bold]Welcome to ReAgt[/bold] - Your Adaptive Training Coach", style="blue")
    )

    # Sports
    console.print("\nAvailable sports: " + ", ".join(AVAILABLE_SPORTS))
    sports_input = Prompt.ask(
        "What sport(s) do you train?",
        default="running",
    )
    sports = [s.strip().lower() for s in sports_input.split(",")]

    # Goal event
    event = Prompt.ask("What's your goal event?", default="Half Marathon")

    # Target date
    target_date = Prompt.ask("Target date (YYYY-MM-DD)?", default="2026-08-15")

    # Target time
    target_time = Prompt.ask("Target time (H:MM:SS)?", default="1:45:00")

    # Training days
    training_days = IntPrompt.ask(
        "How many days per week can you train?", default=5
    )

    # Max session duration
    max_duration = IntPrompt.ask(
        "Max session duration in minutes?", default=90
    )

    profile = create_profile(
        sports=sports,
        event=event,
        target_date=target_date,
        target_time=target_time,
        training_days_per_week=training_days,
        max_session_minutes=max_duration,
    )

    path = save_profile(profile)
    console.print(f"\n[green]Profile saved to {path}[/green]")
    return profile


def display_plan(plan: dict) -> None:
    """Display a training plan as a Rich table."""
    table = Table(title="Weekly Training Plan", show_lines=True)
    table.add_column("Day", style="bold", width=10)
    table.add_column("Sport", width=10)
    table.add_column("Type", style="cyan", width=16)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Description", width=40)
    table.add_column("Notes", width=25)

    for session in plan.get("sessions", []):
        table.add_row(
            session.get("day", ""),
            session.get("sport", ""),
            session.get("type", ""),
            f"{session.get('duration_minutes', '?')} min",
            session.get("description", ""),
            session.get("notes", ""),
        )

    console.print(table)

    summary = plan.get("weekly_summary", {})
    if summary:
        console.print(
            Panel(
                f"Total sessions: {summary.get('total_sessions', '?')} | "
                f"Total duration: {summary.get('total_duration_minutes', '?')} min | "
                f"Focus: {summary.get('focus', 'N/A')}",
                title="Weekly Summary",
                style="green",
            )
        )


def main():
    """Main CLI entry point: onboard -> generate plan -> display."""
    profile = onboard_athlete()

    console.print("\n[yellow]Generating your training plan...[/yellow]\n")

    try:
        plan = generate_plan(profile)
    except ValueError as e:
        console.print(f"[red]Error generating plan: {e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Failed to connect to Gemini: {e}[/red]")
        return

    path = save_plan(plan)
    console.print(f"[green]Plan saved to {path}[/green]\n")

    display_plan(plan)


if __name__ == "__main__":
    main()
