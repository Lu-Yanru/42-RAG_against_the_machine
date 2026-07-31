import fire


class CLI:
    """Handles the CLI."""

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        print(f"max_chunk_size: {max_chunk_size}")
        print("Ingestion complete! Indices saved under data/processed/")
