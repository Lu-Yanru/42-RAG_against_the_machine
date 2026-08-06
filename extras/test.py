from mrkdwn_analysis import MarkdownAnalyzer


analyzer = MarkdownAnalyzer("data/raw/vllm-0.10.1/RELEASE.md")

print(analyzer.identify_paragraphs())
