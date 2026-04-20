#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AMS - Activation-based Model Scanner

Command-line interface for scanning language models for safety degradation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Rich console for beautiful output
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def print_banner():
    """Print AMS banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║     █████╗ ███╗   ███╗███████╗                                ║
║    ██╔══██╗████╗ ████║██╔════╝                                ║
║    ███████║██╔████╔██║███████╗                                ║
║    ██╔══██║██║╚██╔╝██║╚════██║                                ║
║    ██║  ██║██║ ╚═╝ ██║███████║                                ║
║    ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝                                ║
║                                                               ║
║    Activation-based Model Scanner v0.1.0                      ║
║    Verify model safety via activation pattern analysis        ║
╚═══════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(banner, style="cyan")
    else:
        print(banner)


def format_result_rich(
    safety_report,
    verification_report=None,
    console=None,
):
    """Format scan results with rich formatting."""
    if console is None:
        console = Console()

    # Tier 1 Results
    console.print("\n[bold blue]Tier 1: Generic Safety Check[/bold blue]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Concept", style="cyan")
    table.add_column("Separation", justify="right")
    table.add_column("Level", justify="center")
    table.add_column("Status", justify="center")
    if safety_report.comparison_baseline:
        table.add_column("vs Baseline", justify="right")

    level_colors = {"CRITICAL": "red", "WARNING": "yellow", "PASS": "green"}

    for name, result in safety_report.concept_results.items():
        level_color = level_colors.get(result.safety_level, "white")
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"

        row = [
            name,
            f"[{level_color}]{result.separation:.2f}σ[/{level_color}]",
            f"[{level_color}]{result.safety_level}[/{level_color}]",
            status,
        ]

        if safety_report.comparison_baseline and result.drift_percent is not None:
            drift_color = (
                "green"
                if result.drift_percent >= -10
                else ("yellow" if result.drift_percent >= -30 else "red")
            )
            row.append(f"[{drift_color}]{result.drift_percent:+.1f}%[/{drift_color}]")
        elif safety_report.comparison_baseline:
            row.append("-")

        table.add_row(*row)

    console.print(table)

    # Overall result with level
    level_color = level_colors.get(safety_report.overall_level, "white")
    if safety_report.overall_level == "PASS":
        console.print(
            Panel(
                f"[green bold]Overall: {safety_report.overall_level}[/green bold]\n\n"
                f"{safety_report.recommendation}",
                title="Tier 1 Result",
                border_style="green",
            )
        )
    elif safety_report.overall_level == "WARNING":
        console.print(
            Panel(
                f"[yellow bold]Overall: {safety_report.overall_level}[/yellow bold]\n\n"
                f"{safety_report.recommendation}",
                title="Tier 1 Result",
                border_style="yellow",
            )
        )
    else:  # CRITICAL
        console.print(
            Panel(
                f"[red bold]Overall: {safety_report.overall_level}[/red bold]\n\n"
                f"{safety_report.recommendation}",
                title="Tier 1 Result",
                border_style="red",
            )
        )

    # Tier 2 Results (if present)
    if verification_report is not None:
        console.print("\n[bold blue]Tier 2: Identity Verification[/bold blue]")
        console.print(f"Claimed: [cyan]{verification_report.claimed_identity}[/cyan]")

        if verification_report.verified is None:
            console.print(f"[yellow]{verification_report.reason}[/yellow]")
        else:
            table2 = Table(show_header=True, header_style="bold")
            table2.add_column("Concept", style="cyan")
            table2.add_column("Direction Similarity", justify="right")
            table2.add_column("Separation Drift", justify="right")
            table2.add_column("Status", justify="center")

            for check in verification_report.checks:
                status = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
                sim_style = "green" if check.direction_similarity >= 0.8 else "red"
                drift_style = "green" if check.separation_drift <= 0.2 else "red"
                table2.add_row(
                    check.concept,
                    f"[{sim_style}]{check.direction_similarity:.2f}[/{sim_style}]",
                    f"[{drift_style}]{check.separation_drift*100:.1f}%[/{drift_style}]",
                    status,
                )

            console.print(table2)

            if verification_report.verified:
                console.print(
                    Panel(
                        "[green bold]Identity: VERIFIED ✓[/green bold]",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[red bold]Identity: NOT VERIFIED ✗[/red bold]\n\n"
                        f"{verification_report.reason}",
                        border_style="red",
                    )
                )

    # Scan metadata
    console.print(f"\n[dim]Scan completed in {safety_report.scan_time:.1f}s[/dim]")


def format_result_plain(safety_report, verification_report=None):
    """Format scan results as plain text."""
    lines = []
    lines.append("\nTier 1: Generic Safety Check")
    lines.append("-" * 60)

    for name, result in safety_report.concept_results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        drift_str = ""
        if result.drift_percent is not None:
            drift_str = f" ({result.drift_percent:+.1f}% vs baseline)"
        lines.append(
            f"  {name}: {result.separation:.2f}σ [{result.safety_level}] {status}{drift_str}"
        )

    lines.append("")
    lines.append(f"Overall: {safety_report.overall_level}")
    lines.append(f"Recommendation: {safety_report.recommendation}")

    if verification_report is not None:
        lines.append("\nTier 2: Identity Verification")
        lines.append("-" * 60)
        lines.append(f"Claimed: {verification_report.claimed_identity}")

        if verification_report.verified is None:
            lines.append(verification_report.reason)
        else:
            for check in verification_report.checks:
                status = "✓" if check.passed else "✗"
                lines.append(f"  {check.concept}:")
                lines.append(f"    Direction similarity: {check.direction_similarity:.2f} {status}")
                lines.append(f"    Separation drift: {check.separation_drift*100:.1f}%")

            verified = "VERIFIED ✓" if verification_report.verified else "NOT VERIFIED ✗"
            lines.append(f"\nIdentity: {verified}")
            if verification_report.reason:
                lines.append(verification_report.reason)

    lines.append(f"\nScan completed in {safety_report.scan_time:.1f}s")

    return "\n".join(lines)


def cmd_scan(args):
    """Execute scan command."""
    from .scanner import ModelScanner

    if not args.quiet:
        print_banner()

    scanner = ModelScanner(
        baselines_dir=args.baselines_dir,
        device=args.device,
        dtype=args.dtype,
    )

    console = Console() if RICH_AVAILABLE else None

    # Show progress
    try:
        if RICH_AVAILABLE and not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_desc = "Loading model..."
                if args.compare:
                    task_desc = f"Loading models (comparing to {args.compare})..."
                progress.add_task(task_desc, total=None)
                safety_report, verification_report = scanner.full_scan(
                    model_path=args.model,
                    claimed_identity=args.verify,
                    mode=args.mode,
                    batch_size=args.batch_size,
                    compare_to=args.compare,
                    concepts_file=args.concepts_file,
                    trust_remote_code=args.trust_remote_code,
                    load_in_8bit=args.load_8bit,
                    load_in_4bit=args.load_4bit,
                )
        else:
            if not args.quiet:
                print("Loading model...")
            safety_report, verification_report = scanner.full_scan(
                model_path=args.model,
                claimed_identity=args.verify,
                mode=args.mode,
                batch_size=args.batch_size,
                compare_to=args.compare,
                concepts_file=args.concepts_file,
                trust_remote_code=args.trust_remote_code,
                load_in_8bit=args.load_8bit,
                load_in_4bit=args.load_4bit,
            )
    except ValueError as e:
        if RICH_AVAILABLE and not args.quiet:
            console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        else:
            print(f"\nConfiguration Error: {e}")
        sys.exit(1)

    # Output results
    if args.json:
        output = {
            "safety_report": safety_report.to_dict(),
            "verification_report": verification_report.to_dict() if verification_report else None,
        }
        print(json.dumps(output, indent=2))
    elif RICH_AVAILABLE and not args.quiet:
        format_result_rich(safety_report, verification_report, console)
    else:
        print(format_result_plain(safety_report, verification_report))

    # Exit code based on safety level
    if safety_report.overall_level == "CRITICAL":
        sys.exit(1)
    if safety_report.overall_level == "WARNING":
        sys.exit(2)
    if verification_report and verification_report.verified is False:
        sys.exit(3)


def cmd_baseline(args):
    """Execute baseline command."""
    from .scanner import ModelScanner

    scanner = ModelScanner(
        baselines_dir=args.baselines_dir,
        device=args.device,
        dtype=args.dtype,
    )

    if args.action == "create":
        print(f"Creating baseline for {args.model}...")
        baseline = scanner.create_baseline(
            model_path=args.model,
            model_id=args.model_id or args.model,
            mode=args.mode,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            load_in_8bit=args.load_8bit,
            load_in_4bit=args.load_4bit,
        )
        print(f"Baseline created and saved for: {baseline.model_id}")
        print(f"Concepts: {list(baseline.directions.keys())}")
        print(f"Separations: {baseline.separations}")

    elif args.action == "list":
        baselines = scanner.baselines_db.list_baselines()
        if baselines:
            print("Available baselines:")
            for b in baselines:
                print(f"  - {b}")
        else:
            print("No baselines found.")

    elif args.action == "show":
        if not args.model_id:
            print("Error: --model-id required for 'show' action")
            sys.exit(1)
        baseline = scanner.baselines_db.get_baseline(args.model_id)
        if baseline is None:
            print(f"No baseline found for: {args.model_id}")
            sys.exit(1)
        print(
            json.dumps(
                {
                    "model_id": baseline.model_id,
                    "separations": baseline.separations,
                    "optimal_layers": baseline.optimal_layers,
                    "model_info": baseline.model_info,
                    "created_at": baseline.created_at,
                },
                indent=2,
            )
        )


def cmd_concepts(args):
    """Show available safety concepts."""
    from .concepts import UNIVERSAL_SAFETY_CHECKS

    if args.json:
        import tempfile

        from .concepts import export_concepts_to_json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_concepts_to_json(f.name)
            with open(f.name) as rf:
                print(rf.read())
    else:
        print("\nAvailable Safety Concepts:")
        print("=" * 60)
        for name, concept in UNIVERSAL_SAFETY_CHECKS.items():
            print(f"\n{name}")
            print(f"  Description: {concept.description}")
            print(f"  Pairs: {len(concept.pairs)}")
            print(f"  Min separation: {concept.min_separation}σ")
            if args.verbose:
                print("  Sample pairs:")
                for i, pair in enumerate(concept.pairs[:3]):
                    print(f'    {i+1}. + "{pair.positive[:50]}..."')
                    print(f'       - "{pair.negative[:50]}..."')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AMS - Activation-based Model Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic safety scan
  ams scan meta-llama/Llama-3-8B-Instruct

  # Scan with identity verification
  ams scan ./my-model --verify meta-llama/Llama-3-8B-Instruct

  # Quick scan (fewer concepts, faster)
  ams scan ./model --mode quick

  # JSON output for CI/CD
  ams scan ./model --json

  # Create baseline for a model
  ams baseline create meta-llama/Llama-3-8B-Instruct

  # List available baselines
  ams baseline list
        """,
    )

    # Global options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument("--dtype", default="float16", help="Data type (float16/float32/bfloat16)")
    parser.add_argument("--baselines-dir", default="./baselines", help="Baselines directory")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a model for safety")
    scan_parser.add_argument("model", help="Model path or HuggingFace ID")
    scan_parser.add_argument(
        "--verify", "-V", metavar="MODEL_ID", help="Verify identity against baseline"
    )
    scan_parser.add_argument(
        "--compare", "-C", metavar="MODEL_ID", help="Compare separations against another model"
    )
    scan_parser.add_argument(
        "--mode", choices=["quick", "standard", "full"], default="standard", help="Scan mode"
    )
    scan_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for activation extraction"
    )
    scan_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    scan_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code in model"
    )
    scan_parser.add_argument(
        "--load-8bit", action="store_true", help="Load model in 8-bit quantization"
    )
    scan_parser.add_argument(
        "--load-4bit", action="store_true", help="Load model in 4-bit quantization"
    )
    scan_parser.add_argument(
        "--concepts-file", metavar="FILE", help="Custom JSON file with safety concepts"
    )
    scan_parser.set_defaults(func=cmd_scan)

    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Manage baselines")
    baseline_parser.add_argument(
        "action", choices=["create", "list", "show"], help="Baseline action"
    )
    baseline_parser.add_argument("model", nargs="?", help="Model to create baseline for")
    baseline_parser.add_argument("--model-id", help="Custom model ID for baseline")
    baseline_parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Concepts to include",
    )
    baseline_parser.add_argument("--batch-size", type=int, default=8)
    baseline_parser.add_argument("--trust-remote-code", action="store_true")
    baseline_parser.add_argument("--load-8bit", action="store_true")
    baseline_parser.add_argument("--load-4bit", action="store_true")
    baseline_parser.set_defaults(func=cmd_baseline)

    # Concepts command
    concepts_parser = subparsers.add_parser("concepts", help="Show safety concepts")
    concepts_parser.add_argument("--json", action="store_true", help="Output as JSON")
    concepts_parser.add_argument("-v", "--verbose", action="store_true", help="Show sample pairs")
    concepts_parser.set_defaults(func=cmd_concepts)

    args = parser.parse_args()

    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
