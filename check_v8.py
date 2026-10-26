"""Static safety checks; run from ZIP root with: python tests/check_v8.py"""
from pathlib import Path
import ast
src=Path(__file__).resolve().parents[1]/'222.py'
text=src.read_text(encoding='utf-8')
tree=ast.parse(text)
defs={node.name:node.lineno for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert defs['add_hand_temporal_notes'] < text[:text.find('df_result = add_hand_temporal_notes')].count('\n')+1
for name in ('v8_neck_candidate','v8_neck_cross_view','v8_show_rejected_center','v8_show_neck_center','v8_download_report_zip'):
 assert name in defs,name
assert 'RULA_V8_WIZARD_REJECT_NECK_2026-10-27' in text
assert '雙側同分' in text
compile(text,str(src),'exec')
print('RULA v8 syntax, helper ordering, version and page checks PASS')
