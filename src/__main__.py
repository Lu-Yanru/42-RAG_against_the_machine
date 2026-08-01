import fire

from src.cli import CLI


def main() -> None:
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
