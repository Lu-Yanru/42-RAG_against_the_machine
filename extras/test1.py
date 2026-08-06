import ast


tree = ast.parse("src/ingest/chunking.py")
print(ast.dump(tree, indent=4))
body = getattr(tree, "body", [])
print(type(body[0]))
