from src.cli import CLI
import fire


def main() -> None:
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("Interrputed by user.")


if __name__ == "__main__":
    main()
