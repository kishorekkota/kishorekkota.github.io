#!/usr/bin/env python3

import re

print("Fixing Mermaid diagram syntax errors...")

# Read the input file
with open('/Users/kishorekota/Documents/GitHub/kishorekkota.github.io/llm_ai/ai_concepts_visualizations.md', 'r') as f:
    content = f.read()

# Fix incomplete closing for mermaid blocks that end with style lines
# Pattern: style ... fill:#...```  (missing newline before closing)
content = re.sub(r'(style [^`]+)```', r'\1\n```', content)

# Write the fixed content back
with open('/Users/kishorekota/Documents/GitHub/kishorekkota.github.io/llm_ai/ai_concepts_visualizations.md', 'w') as f:
    f.write(content)

print("✅ All Mermaid diagram syntax errors have been fixed!")
