from dotenv import load_dotenv
import fire
import os

from src.cli import CLI


def main() -> None:
    # Load HF token
    if os.path.exists(".env"):
        load_dotenv()

    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
