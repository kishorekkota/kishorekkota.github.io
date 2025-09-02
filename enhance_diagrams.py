#!/usr/bin/env python3

import re
import sys

print("Enhancing Mermaid diagrams for better readability...")

# Read the input file
with open('/Users/kishorekota/Documents/GitHub/kishorekkota.github.io/llm_ai/ai_concepts_visualizations.md', 'r') as f:
    content = f.read()

# Function to enhance a Mermaid diagram
def enhance_diagram(match):
    diagram_content = match.group(1)
    
    # Skip if already enhanced (has init config)
    if '%%{init:' in diagram_content:
        return match.group(0)
    
    # Add the sizing configuration
    enhanced = f'''```mermaid
%%{{init: {{'theme':'base', 'themeVariables': {{ 'primaryColor': '#ffffff', 'primaryTextColor': '#333333', 'primaryBorderColor': '#cccccc', 'lineColor': '#666666'}}}}}}%%
{diagram_content}```'''
    
    return enhanced

# Pattern to match Mermaid code blocks
pattern = r'```mermaid\n(.*?)\n```'

# Replace all Mermaid diagrams with enhanced versions
enhanced_content = re.sub(pattern, enhance_diagram, content, flags=re.DOTALL)

# Write the enhanced content back
with open('/Users/kishorekota/Documents/GitHub/kishorekkota.github.io/llm_ai/ai_concepts_visualizations.md', 'w') as f:
    f.write(enhanced_content)

print("✅ All Mermaid diagrams have been enhanced with better sizing and styling!")
