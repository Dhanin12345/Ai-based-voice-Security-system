import os
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def get_python_executable():
    """
    Finds virtualenv Python executable if available, otherwise fallback to sys.executable.
    """
    venv_win = BASE_DIR / "venv" / "Scripts" / "python.exe"
    venv_unix = BASE_DIR / "venv" / "bin" / "python"
    
    if venv_win.exists():
        return str(venv_win)
    elif venv_unix.exists():
        return str(venv_unix)
    return sys.executable

def run_update_data(samples=25, clean=False, model="rf"):
    """
    Executes the data update pipeline (audio generation, feature extraction, ML training).
    """
    py = get_python_executable()
    cmd = [py, "ml/update_data.py", "--samples", str(samples), "--model", model]
    if clean:
        cmd.append("--clean")
    print(f"\n[PIPELINE] Running data update: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

def run_status():
    """
    Shows current dataset and model status.
    """
    py = get_python_executable()
    subprocess.run([py, "ml/update_data.py", "--status"], cwd=str(BASE_DIR), check=True)

def run_tests():
    """
    Runs pytest test suite.
    """
    py = get_python_executable()
    print(f"\n[PIPELINE] Running test suite with {py} -m pytest tests/ -v")
    subprocess.run([py, "-m", "pytest", "tests/", "-v"], cwd=str(BASE_DIR), check=True)

def run_server(host="0.0.0.0", port=8000, reload=True):
    """
    Launches the FastAPI server and dashboard.
    """
    py = get_python_executable()
    cmd = [py, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    print("\n" + "=" * 60)
    print("      AI VOICE CLONING DETECTION SERVER STARTING")
    print("=" * 60)
    print(f"  • Web Dashboard URL : http://localhost:{port}/dashboard.html")
    print(f"  • Overview Page URL : http://localhost:{port}/index.html")
    print(f"  • History Logs URL  : http://localhost:{port}/history.html")
    print(f"  • Interactive Docs  : http://localhost:{port}/docs")
    print("=" * 60 + "\n")
    
    subprocess.run(cmd, cwd=str(BASE_DIR))

def main():
    parser = argparse.ArgumentParser(
        description="VoiceGuard Master Data & Application Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python run_pipeline.py --update-data          # Update audio, extract features, train ML model
  python run_pipeline.py --status               # Check dataset counts and model metrics
  python run_pipeline.py --server               # Launch FastAPI backend & web dashboard
  python run_pipeline.py --test                 # Run pytest unit and integration tests
  python run_pipeline.py --all                  # Update data, run tests, and start server
        """
    )

    parser.add_argument("--update-data", action="store_true", help="Generate/update dataset, extract features & train model")
    parser.add_argument("--status", action="store_true", help="View current dataset and trained model status")
    parser.add_argument("--server", action="store_true", help="Run the FastAPI web server & dashboard")
    parser.add_argument("--test", action="store_true", help="Run the test suite")
    parser.add_argument("--all", action="store_true", help="Update data, verify tests, and start server")
    
    parser.add_argument("--samples", type=int, default=25, help="Number of audio samples per class (default: 25)")
    parser.add_argument("--clean", action="store_true", help="Clean existing WAV files before generating new ones")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm", "gb"], help="Model type: rf, svm, gb")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the web server on (default: 8000)")

    args = parser.parse_args()

    # If no flags provided, show help and status
    if not (args.update_data or args.status or args.server or args.test or args.all):
        parser.print_help()
        print("\nShowing current status:")
        run_status()
        return

    if args.status:
        run_status()

    if args.update_data or args.all:
        run_update_data(samples=args.samples, clean=args.clean, model=args.model)

    if args.test or args.all:
        run_tests()

    if args.server or args.all:
        run_server(port=args.port)

if __name__ == "__main__":
    main()
