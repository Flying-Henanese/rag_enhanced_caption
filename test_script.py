import re
text = "### 高性能文档解析方案|核心技术亮点|2. 智能表格处理"
first_line = text.split('\n')[0].strip()
match = re.match(r'^#+\s+(.*)', first_line)
if match:
    levels = match.group(1).split('|')
    print(levels)
