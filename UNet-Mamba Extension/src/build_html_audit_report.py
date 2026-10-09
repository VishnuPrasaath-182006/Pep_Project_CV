"""
Converts CAVE_Flood_Final_Independent_Audit_Report.md into a publication-grade HTML document.
Generates CAVE_Flood_Final_Independent_Audit_Report.html with clean responsive styling.
"""

import re
from pathlib import Path


def format_inline(text):
    # LaTeX math replacement
    text = re.sub(r"\$\$(.*?)\$\$", r'<span class="math-display">\1</span>', text)
    text = re.sub(r"\$(.*?)\$", r'<span class="math-inline">\1</span>', text)
    # Bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    # Inline code
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    # Badges
    if text.strip() == "PASS":
        return '<span class="badge badge-pass">PASS</span>'
    if text.strip() == "VALID WITH LIMITATIONS":
        return '<span class="badge badge-verdict">VALID WITH LIMITATIONS</span>'
    return text


def md_to_html(md_text):
    html_lines = []
    in_table = False
    in_code = False
    table_header = True
    
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Code block
        if line.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code = True
            i += 1
            continue
            
        if in_code:
            escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(escaped)
            i += 1
            continue
            
        # Table
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            
            # Check if separator row
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                i += 1
                table_header = False
                continue
                
            if not in_table:
                html_lines.append('<div class="table-container"><table>')
                in_table = True
                table_header = True
                
            tag = "th" if table_header else "td"
            row_html = "<tr>" + "".join(f"<{tag}>{format_inline(c)}</{tag}>" for c in cells) + "</tr>"
            html_lines.append(row_html)
            i += 1
            continue
        else:
            if in_table:
                html_lines.append("</table></div>")
                in_table = False
                
        # Empty line
        if not line.strip():
            html_lines.append("")
            i += 1
            continue
            
        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{format_inline(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{format_inline(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{format_inline(line[4:].strip())}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{format_inline(line[5:].strip())}</h4>")
        elif line.startswith("- "):
            html_lines.append(f"<ul><li>{format_inline(line[2:].strip())}</li></ul>")
        elif re.match(r"^\d+\.\s", line):
            content = re.sub(r"^\d+\.\s", "", line)
            html_lines.append(f"<ol><li>{format_inline(content.strip())}</li></ol>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        else:
            html_lines.append(f"<p>{format_inline(line.strip())}</p>")
            
        i += 1
        
    if in_table:
        html_lines.append("</table></div>")
        
    raw_html = "\n".join(html_lines)
    raw_html = re.sub(r"</ul>\s*<ul>", "", raw_html)
    raw_html = re.sub(r"</ol>\s*<ol>", "", raw_html)
    return raw_html


def build_audit_html():
    cur_file = Path(__file__).resolve()
    base_dir = cur_file.parent.parent
    ws_dir = base_dir.parent
        
    candidates = [
        base_dir / "results" / "final" / "CAVE_Flood_Final_Report_v3_CORRECTED.md",
        ws_dir / "UNet-Mamba Extension" / "results" / "final" / "CAVE_Flood_Final_Report_v3_CORRECTED.md",
        ws_dir / "UNet-Mamba" / "results" / "final" / "CAVE_Flood_Final_Report_v3_CORRECTED.md",
        ws_dir / "README.md"
    ]
    md_path = next((p for p in candidates if p.exists()), candidates[0])
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    body_html = md_to_html(md_text)
    
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAVE-Flood v1: Final Independent Verification & Audit Report</title>
<style>
  :root {{
    --primary: #0284c7;
    --primary-dark: #0369a1;
    --text: #0f172a;
    --text-muted: #475569;
    --bg: #ffffff;
    --bg-alt: #f8fafc;
    --border: #e2e8f0;
    --success: #15803d;
    --warning: #b45309;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: var(--text);
    background-color: var(--bg-alt);
    margin: 0;
    padding: 32px 16px;
  }}
  .container {{
    max-width: 1100px;
    margin: 0 auto;
    background: var(--bg);
    padding: 48px;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
    border: 1px solid var(--border);
  }}
  h1 {{
    font-size: 26px;
    color: #0f172a;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
    margin-top: 0;
    line-height: 1.25;
  }}
  h2 {{
    font-size: 20px;
    color: #1e293b;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-top: 36px;
  }}
  h3 {{
    font-size: 16px;
    color: #334155;
    margin-top: 24px;
  }}
  p, li {{
    font-size: 14.5px;
    color: #334155;
  }}
  ul, ol {{
    padding-left: 24px;
    margin-bottom: 16px;
  }}
  li {{
    margin-bottom: 6px;
  }}
  .table-container {{
    overflow-x: auto;
    margin: 20px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: #ffffff;
    border: 1px solid var(--border);
  }}
  th, td {{
    padding: 10px 12px;
    text-align: left;
    border: 1px solid var(--border);
  }}
  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
  }}
  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
  pre {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.45;
  }}
  code {{
    font-family: Consolas, Monaco, "Courier New", monospace;
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 12.5px;
  }}
  pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
  }}
  .badge {{
    display: inline-block;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .badge-pass {{
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
  }}
  .badge-verdict {{
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
  }}
  hr {{
    border: 0;
    height: 1px;
    background: var(--border);
    margin: 32px 0;
  }}
</style>
</head>
<body>
<div class="container">
{body_html}
</div>
</body>
</html>
"""
    out_path = base_dir / "results" / "final" / "CAVE_Flood_Final_Independent_Audit_Report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Generated HTML report: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_audit_html()
