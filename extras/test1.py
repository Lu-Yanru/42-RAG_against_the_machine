import ast


file_path = "src/ingest/chunking_text.py"
try:
	with open(file_path, "r", encoding="utf-8") as fh:
		source = fh.read()
except FileNotFoundError:
	raise SystemExit(f"File not found: {file_path}")

tree = ast.parse(source, filename=file_path)
# print(ast.dump(tree, indent=4))
print(tree.body[5].end_col_offset)
