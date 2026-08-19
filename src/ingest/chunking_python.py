import ast
import sys
from typing import TypeGuard
from tqdm import tqdm

from src.ingest.chunking import Chunk, Chunker
from src.ingest.loader import File


class PythonChunker(Chunker):

    _DefOrClassNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

    def __init__(self, files: list[File],
                 max_chunk_size: int = 2000,
                 file_type: str = "py") -> None:
        super().__init__(files, max_chunk_size, file_type)

    def chunk(self) -> list[Chunk]:
        chunks = []
        for file in tqdm(self.files, desc="Chunking py files"):
            if not file.content.strip():
                continue
            try:
                tree = ast.parse(file.content, filename=file.file_path)
            except SyntaxError as e:
                print(f"Error parsing AST of file '{file.file_path}': ",
                      f"{e} Skipping...", file=sys.stderr)
                continue

            units = PythonChunker._member_unit_span(tree.body, 0,
                                                    len(file.content),
                                                    file.line_offsets)
            class_contexts = PythonChunker._class_context(tree,
                                                          file.line_offsets)
            for u_start, u_end in units:
                spans = self.chunk_span(file.content, start=u_start,
                                        end=u_end,
                                        file_path=file.file_path)
                context = PythonChunker._enclosing_class(
                    u_start, u_end, class_contexts
                ) or ""

                for start, end in spans:
                    chunks.append(Chunk(
                        file_path=file.file_path,
                        first_character_index=start,
                        last_character_index=end,
                        content=file.content[start:end],
                        context=context,
                    ))
        return chunks

    @staticmethod
    def _is_def(node: ast.AST) \
            -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
        """
        Check if the ast node is a function definition.
        """
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    @staticmethod
    def _is_class(node: ast.AST) -> TypeGuard[ast.ClassDef]:
        """
        Check if the ast node is a class definition.
        """
        return isinstance(node, ast.ClassDef)

    @staticmethod
    def _char_offset(line_offsets: list[int], lineno: int | None,
                     col: int | None) -> int:
        """Caculates the absolute offset of a character."""
        if lineno is None:
            lineno = 0
        if col is None:
            col = 0
        return line_offsets[lineno - 1] + col

    @staticmethod
    def _def_start(node: _DefOrClassNode, line_offsets: list[int]) -> int:
        """
        Calculates the absolute start offset of a function/class,
        including its decorator.
        """
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            first = decorators[0]
            return PythonChunker._char_offset(line_offsets, first.lineno,
                                              first.col_offset - 1)

        return PythonChunker._char_offset(line_offsets, node.lineno,
                                          node.col_offset)

    @staticmethod
    def _def_end(node: _DefOrClassNode, line_offsets: list[int]) -> int:
        """
        Calculates the absolut offset of the end of a function/class.
        """
        return PythonChunker._char_offset(line_offsets, node.end_lineno,
                                          node.end_col_offset)

    @staticmethod
    def _member_unit_span(members: list[ast.stmt], region_start: int,
                          region_end: int,
                          line_offsets: list[int]) -> list[tuple[int, int]]:
        """
        Given an ordered list of statements (a module body or a class
        body) and the (region_start, region_end) they live in, return unit
        spans: each function/method is its own unit; each class recurses
        into header + its own members (so nested classes split cleanly too,
        at any depth); everything else (imports, assignments, bare
        expressions, if-blocks, ...) is folded into a "preamble" span that
        runs from wherever the previous unit left off up to the next
        function/class boundary.
        """
        units = []
        cursor = region_start

        for member in members:
            if PythonChunker._is_def(member):
                member_start = PythonChunker._def_start(member, line_offsets)

                # Save preamble or what is in between functions
                if member_start > cursor:
                    units.append((cursor, member_start))

                member_end = PythonChunker._def_end(member, line_offsets)
                units.append((member_start, member_end))
                cursor = member_end

            elif PythonChunker._is_class(member):
                class_start = PythonChunker._def_start(member, line_offsets)
                if class_start > cursor:
                    units.append((cursor, class_start))
                # Recursively split in classes and functions
                units.extend(PythonChunker._class_unit_span(member,
                                                            line_offsets))
                cursor = PythonChunker._def_end(member, line_offsets)
            # else: If it is an ordinary statement,
            # group it together with the next function/class def.

        # save the rest of the file until the EOF
        if region_end > cursor:
            units.append((cursor, region_end))

        return units

    @staticmethod
    def _class_unit_span(node: ast.ClassDef,
                         line_offsets: list[int]) -> list[tuple[int, int]]:
        start = PythonChunker._def_start(node, line_offsets)
        end = PythonChunker._def_end(node, line_offsets)
        return PythonChunker._member_unit_span(node.body, start,
                                               end, line_offsets)

    @staticmethod
    def _class_context(tree: ast.Module,
                       line_offsets: list[int]) -> list[tuple[int, int, str]]:
        """
        Returns (class_start, class_end, qualified_name) for every
        ClassDef in the module, at any nesting depth, qualified_name
        dot-joined outer-to-inner (e.g. "Outer.Inner"). Does not
        recurse into function bodies.
        """
        contexts: list[tuple[int, int, str]] = []

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if PythonChunker._is_class(child):
                    start = PythonChunker._def_start(child, line_offsets)
                    end = PythonChunker._def_end(child, line_offsets)
                    qualified = (f"{prefix}.{child.name}" if prefix
                                 else child.name)
                    contexts.append((start, end, qualified))
                    walk(child, qualified)
                elif PythonChunker._is_def(child):
                    continue
                else:
                    walk(child, prefix)

        walk(tree, "")
        return contexts

    @staticmethod
    def _enclosing_class(start: int, end: int,
                         class_contexts: list[tuple[int, int, str]]) \
            -> str | None:
        """Innermost class whose span fully contains [start, end)."""
        candidates = [(c_end - c_start, name)
                      for c_start, c_end, name in class_contexts
                      if c_start <= start and end <= c_end]
        # Chunk is not inside any class
        if not candidates:
            return None
        # Get the name of the inner most class
        return min(candidates)[1]
