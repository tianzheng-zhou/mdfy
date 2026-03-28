"""临时测试：跨页断句检测与合并"""
from pdf_to_md_ai import _is_incomplete_sentence, _is_continuation_start, _join_pages_smart

# Test real patterns from the output file
print("=== _is_incomplete_sentence ===")
tests = [
    ('应该更具概括性', True),   # incomplete
    ('讨论语', True),           # word break
    ('怎能确', True),           # word break  
    ('可能会被他们', True),     # incomplete
    ('因此，有时在', True),     # incomplete
    ('研究方', True),           # word break
    ('言语行为的复杂形', True), # word break
    ('这种有容乃大的精神值得我们学习。', False),  # complete
    ('## 第一章 语言研究', False),  # heading
    ('![图1](images/page1.png)', False),  # image
    ('和普遍性，应该在多种语言中比较、归纳和抽象。', False),  # complete
]

all_pass = True
for text, expected in tests:
    result = _is_incomplete_sentence(text)
    status = 'OK' if result == expected else 'FAIL'
    if result != expected:
        all_pass = False
    print(f'  {status}: "{text[:30]}" -> {result} (expected {expected})')

print("\n=== _is_continuation_start ===")
cont_tests = [
    ('和普遍性，应该在', True),     # starts with connector 和
    ('言问题，但他', True),          # has comma within first 5 chars
    ('定这些声音片段', False),       # no obvious continuation signal -> rely on prompt
    ('视为一种更加复杂的', False),   # no obvious continuation signal -> rely on prompt
    ('字母组合旁附上音标', False),   # no obvious continuation signal -> rely on prompt
    ('向。当获得学位时', True),      # has 。 within first 5 chars
    ('式并非都在我们', False),       # no obvious continuation signal -> rely on prompt
    ('## 第二章 语言人', False),
    ('## Acknowledgements', False),
    ('![图2](images/page2.png)', False),
]

for text, expected in cont_tests:
    result = _is_continuation_start(text)
    status = 'OK' if result == expected else 'FAIL'
    if result != expected:
        all_pass = False
    print(f'  {status}: "{text[:25]}" -> {result} (expected {expected})')

print("\n=== _join_pages_smart ===")
# Simulate real page breaks from the book
parts = [
    '我们所关心的语言现象或规律应该更具概括性',
    '和普遍性，应该在多种语言中比较、归纳和抽象。当然，我们不可能掌握这么多语言。',
    '## 第二章 语言人\n\n语言学家认为"理智"且"博学"的智人',
    '本书作者更倾向于采用历史的、文化的、社会的视角来观察和讨论语',
    '言问题，但他并没有摒弃心理学家和认知语言学家所做的工作。',
]

result = _join_pages_smart(parts)
print("Result preview:")
for i, line in enumerate(result.split('\n')[:15]):
    print(f"  {i}: {line[:80]}")

# Verify merges happened correctly
assert '概括性和普遍性' in result, "Break 1 not merged"
assert '\n\n## 第二章' in result, "Heading should have paragraph break"
# Break 2 (讨论语/言问题) has comma in first 5 chars -> merged
assert '讨论语言问题' in result or '讨论语\n言问题' in result, "Break 2 should be merged"

print("\n" + ("ALL TESTS PASSED!" if all_pass else "SOME TESTS FAILED!"))
