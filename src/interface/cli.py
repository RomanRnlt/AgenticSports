"""CLI interface for ReAgt using Rich."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape

from src.agent.coach import generate_plan, save_plan
from src.agent.proactive import check_proactive_triggers, format_proactive_message
from src.agent.state_machine import AgentCore
from src.agent.trajectory import assess_trajectory
from src.memory.episodes import list_episodes
from src.memory.profile import create_profile, save_profile, load_profile
from src.memory.user_model import UserModel
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone
from src.tools.activity_store import store_activity, list_activities

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


def run_assessment() -> None:
    """Run a full assessment cycle: compare recent activities against current plan."""
    import json

    try:
        profile = load_profile()
    except FileNotFoundError:
        console.print("[red]No athlete profile found. Run onboarding first.[/red]")
        return

    # Find the latest plan
    plans_dir = Path(__file__).parent.parent.parent / "data" / "plans"
    plan_files = sorted(plans_dir.glob("plan_*.json")) if plans_dir.exists() else []
    if not plan_files:
        console.print("[red]No training plan found. Generate a plan first.[/red]")
        return

    latest_plan = json.loads(plan_files[-1].read_text())
    console.print(f"[yellow]Using plan: {plan_files[-1].name}[/yellow]")

    # Get recent activities
    activities = list_activities()
    if not activities:
        console.print("[red]No activities found. Import some training data first.[/red]")
        return

    console.print(f"[yellow]Analyzing {len(activities)} activities...[/yellow]\n")

    # Run the agent cycle
    agent = AgentCore()
    try:
        result = agent.run_cycle(profile, latest_plan, activities)
    except Exception as e:
        console.print(f"[red]Assessment failed: {e}[/red]")
        return

    # Display assessment
    assessment = result.get("assessment", {}).get("assessment", {})
    console.print(Panel(
        f"Compliance: [cyan]{assessment.get('compliance', 'N/A')}[/cyan]\n"
        f"Fitness trend: [cyan]{assessment.get('fitness_trend', 'N/A')}[/cyan]\n"
        f"Fatigue: [cyan]{assessment.get('fatigue_level', 'N/A')}[/cyan]\n"
        f"Injury risk: [cyan]{assessment.get('injury_risk', 'N/A')}[/cyan]",
        title="Training Assessment",
        style="blue",
    ))

    observations = assessment.get("observations", [])
    if observations:
        console.print("\n[bold]Observations:[/bold]")
        for obs in observations:
            console.print(f"  - {obs}")

    # Display autonomy results
    autonomy = result.get("autonomy_result", {})
    auto_applied = autonomy.get("auto_applied", [])
    proposals = autonomy.get("proposals", [])

    if auto_applied:
        console.print(f"\n[green]Auto-applied adjustments ({len(auto_applied)}):[/green]")
        for adj in auto_applied:
            console.print(f"  - {adj.get('description', '?')}")

    if proposals:
        console.print(f"\n[yellow]Proposed adjustments ({len(proposals)}):[/yellow]")
        for adj in proposals:
            impact = adj.get("classified_impact", adj.get("impact", "?"))
            console.print(f"  [{impact}] {adj.get('description', '?')}")

    # Save adjusted plan
    adjusted_plan = result.get("adjusted_plan")
    if adjusted_plan:
        path = save_plan(adjusted_plan)
        console.print(f"\n[green]Adjusted plan saved: {path}[/green]")
        display_plan(adjusted_plan)


def _load_latest_plan() -> dict | None:
    """Load the most recent training plan from data/plans/."""
    import json as _json
    plans_dir = Path(__file__).parent.parent.parent / "data" / "plans"
    plan_files = sorted(plans_dir.glob("plan_*.json")) if plans_dir.exists() else []
    if not plan_files:
        return None
    return _json.loads(plan_files[-1].read_text())


def run_trajectory() -> None:
    """Show full trajectory assessment."""
    try:
        profile = load_profile()
    except FileNotFoundError:
        console.print("[red]No athlete profile found. Run onboarding first.[/red]")
        return

    plan = _load_latest_plan()
    if not plan:
        console.print("[red]No training plan found.[/red]")
        return

    activities = list_activities()
    episodes = list_episodes()

    console.print("[yellow]Assessing training trajectory...[/yellow]\n")

    try:
        traj = assess_trajectory(profile, activities, episodes, plan)
    except Exception as e:
        console.print(f"[red]Trajectory assessment failed: {e}[/red]")
        return

    # Display trajectory
    goal = traj.get("goal", {})
    console.print(Panel(
        f"Event: [cyan]{goal.get('event', '?')}[/cyan]\n"
        f"Target: [cyan]{goal.get('target_time', '?')}[/cyan] by {goal.get('target_date', '?')}\n"
        f"Weeks remaining: [cyan]{goal.get('weeks_remaining', '?')}[/cyan]",
        title="Goal",
        style="blue",
    ))

    trajectory = traj.get("trajectory", {})
    on_track = trajectory.get("on_track", "unknown")
    status_color = "green" if on_track else "red"
    console.print(Panel(
        f"On track: [{status_color}]{on_track}[/{status_color}]\n"
        f"Predicted time: [cyan]{trajectory.get('predicted_race_time', '?')}[/cyan]\n"
        f"Confidence: [cyan]{traj.get('confidence', '?')}[/cyan]\n"
        f"{traj.get('confidence_explanation', '')}",
        title="Trajectory",
        style=status_color,
    ))

    recommendations = traj.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(f"  - {rec}")

    risks = traj.get("risks", [])
    if risks:
        console.print("\n[bold]Risks:[/bold]")
        for risk in risks:
            console.print(f"  [{risk.get('probability', '?')}] {risk.get('risk', '?')}")
            console.print(f"    Mitigation: {risk.get('mitigation', '?')}")


def run_status() -> None:
    """Quick status check with proactive messages."""
    try:
        profile = load_profile()
    except FileNotFoundError:
        console.print("[red]No athlete profile found.[/red]")
        return

    plan = _load_latest_plan()
    activities = list_activities()
    episodes = list_episodes()

    console.print(Panel(
        f"Athlete: [cyan]{profile.get('name', 'Unknown')}[/cyan]\n"
        f"Goal: [cyan]{profile.get('goal', {}).get('event', '?')}[/cyan] "
        f"by {profile.get('goal', {}).get('target_date', '?')}\n"
        f"Activities: [cyan]{len(activities)}[/cyan] | "
        f"Episodes: [cyan]{len(episodes)}[/cyan]",
        title="ReAgt Status",
        style="blue",
    ))

    if plan and activities and episodes:
        try:
            traj = assess_trajectory(profile, activities, episodes, plan)
            triggers = check_proactive_triggers(profile, activities, episodes, traj)

            if triggers:
                console.print("\n[bold]Messages for you:[/bold]")
                for trigger in triggers:
                    msg = format_proactive_message(trigger, profile)
                    priority = trigger.get("priority", "low")
                    color = {"high": "red", "medium": "yellow", "low": "green"}.get(priority, "white")
                    console.print(f"  [{color}]{msg}[/{color}]")
        except Exception as e:
            console.print(f"[dim]Could not generate trajectory: {e}[/dim]")


def run_chat() -> None:
    """Interactive chat mode with conversational coaching.

    If no user model exists, starts in onboarding mode.
    If a user model exists, starts in ongoing coaching mode.
    """
    from src.agent.onboarding import OnboardingEngine
    from src.agent.conversation import ConversationEngine

    user_model = UserModel.load_or_create()
    is_new_user = not user_model.structured_core.get("sports")

    if is_new_user:
        # Onboarding mode
        engine = OnboardingEngine(user_model=user_model)
        greeting = engine.start()
        console.print(Panel(escape(greeting), title="ReAgt Coach", style="blue"))

        while True:
            try:
                user_input = Prompt.ask("\n[bold]You[/bold]")
            except (KeyboardInterrupt, EOFError):
                user_input = "exit"

            if user_input.strip().lower() in ("exit", "quit", "q"):
                engine.end_session()
                user_model.save()
                console.print("[dim]Session saved. See you next time![/dim]")
                break

            response = engine.process_message(user_input)
            console.print(Panel(escape(response), title="ReAgt Coach", style="blue"))

            if engine.is_onboarding_complete():
                console.print(
                    "\n[yellow]Great, I have enough to create your first plan![/yellow]"
                )
                profile = user_model.project_profile()
                beliefs = user_model.get_active_beliefs(min_confidence=0.6)
                try:
                    plan = generate_plan(profile, beliefs=beliefs)
                    path = save_plan(plan)
                    console.print(f"[green]Plan saved to {path}[/green]\n")
                    display_plan(plan)
                except Exception as e:
                    console.print(f"[red]Plan generation failed: {e}[/red]")

                engine.end_session()
                user_model.save()
                console.print("[dim]Session saved. See you next time![/dim]")
                break
    else:
        # Ongoing coaching mode
        engine = ConversationEngine(user_model=user_model)
        session_id = engine.start_session()

        # Deliver proactive messages on startup
        plan = _load_latest_plan()
        activities = list_activities()
        episodes = list_episodes()
        if plan and activities and episodes:
            try:
                profile = user_model.project_profile()
                traj = assess_trajectory(profile, activities, episodes, plan)
                triggers = check_proactive_triggers(profile, activities, episodes, traj)
                if triggers:
                    console.print(Panel(
                        "\n".join(
                            format_proactive_message(t, profile)
                            for t in triggers
                        ),
                        title="Notifications",
                        style="yellow",
                    ))
            except Exception:
                pass  # proactive check is best-effort

        console.print(
            Panel("Welcome back! What's on your mind?", title="ReAgt Coach", style="blue")
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold]You[/bold]")
            except (KeyboardInterrupt, EOFError):
                user_input = "exit"

            if user_input.strip().lower() in ("exit", "quit", "q"):
                engine.end_session()
                user_model.save()
                console.print("[dim]Session saved. See you next time![/dim]")
                break

            response = engine.process_message(user_input)
            console.print(Panel(escape(response), title="ReAgt Coach", style="blue"))


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
    parser.add_argument(
        "--assess", action="store_true",
        help="Run assessment on latest activities vs current plan",
    )
    parser.add_argument(
        "--trajectory", action="store_true",
        help="Show full trajectory assessment",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Quick status check with proactive messages",
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Enter interactive chat mode (default when no flags given)",
    )
    parser.add_argument(
        "--onboard-legacy", action="store_true",
        help="Use legacy form-based onboarding (deprecated)",
    )

    parsed = parser.parse_args(args)

    if parsed.import_file:
        import_activity(parsed.import_file)
        return

    if parsed.assess:
        run_assessment()
        return

    if parsed.trajectory:
        run_trajectory()
        return

    if parsed.status:
        run_status()
        return

    if parsed.onboard_legacy:
        # Legacy form-based onboarding
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
        return

    # Default: chat mode (same as --chat)
    run_chat()


if __name__ == "__main__":
    main()
