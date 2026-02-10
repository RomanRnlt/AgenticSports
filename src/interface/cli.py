"""CLI interface for ReAgt using Rich."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel

from src.agent.coach import generate_plan, save_plan
from src.memory.profile import create_profile, save_profile
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone
from src.tools.activity_store import store_activity

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


def import_activity(file_path: str) -> None:
    """Import a FIT file (or JSON fixture), compute metrics, and store it."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    console.print(f"[yellow]Parsing {path.name}...[/yellow]")
    activity = parse_fit_file(str(path))

    sport = activity.get("sport", "unknown")
    duration_sec = activity.get("duration_seconds", 0)
    duration_min = duration_sec / 60 if duration_sec else 0
    hr = activity.get("heart_rate", {})
    avg_hr = hr.get("avg") if hr else None
    distance = activity.get("distance_meters")

    console.print(f"  Sport: [cyan]{sport}[/cyan]")
    console.print(f"  Duration: [cyan]{duration_min:.0f} min[/cyan]")
    if distance:
        console.print(f"  Distance: [cyan]{distance / 1000:.1f} km[/cyan]")

    # Calculate training load if HR data available
    if avg_hr:
        trimp = calculate_trimp(duration_min, avg_hr)
        zone = classify_hr_zone(avg_hr)
        console.print(f"  Avg HR: [cyan]{avg_hr} bpm[/cyan] (Zone {zone})")
        console.print(f"  TRIMP: [cyan]{trimp}[/cyan]")
        activity["trimp"] = trimp
        activity["hr_zone"] = zone

    stored_path = store_activity(activity)
    console.print(f"\n[green]Activity stored: {stored_path}[/green]")


def main(args: list[str] | None = None):
    """Main CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        prog="reagt",
        description="ReAgt - Adaptive Training Agent for Endurance Athletes",
    )
    parser.add_argument(
        "--import", dest="import_file", metavar="FILE",
        help="Import a FIT file or JSON activity fixture",
    )

    parsed = parser.parse_args(args)

    if parsed.import_file:
        import_activity(parsed.import_file)
        return

    # Default: onboarding flow
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
