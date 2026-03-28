"""Quick test for _normalize_sibling_headings"""
import sys
sys.path.insert(0, '.')
from pdf_to_md_ai import _normalize_sibling_headings

# Test 1: (1) no prefix, (2) has ### → both should get ###
test1 = """一些内容

（1）SPI 事务与 2 个八位字节的头

中间段落文字...

### （2）SPI 事务与一个 3 字节的头部

更多内容"""

result1 = _normalize_sibling_headings(test1)
print("=== Test 1: Mixed levels ===")
for line in result1.split('\n'):
    if '（1）' in line or '（2）' in line:
        print(repr(line))
assert '### （1）SPI' in result1, "FAIL: (1) should get ###"
assert '### （2）SPI' in result1, "FAIL: (2) should keep ###"
print("PASS")

# Test 2: Already consistent → unchanged
test2 = """### （1）标题A

内容

### （2）标题B"""
result2 = _normalize_sibling_headings(test2)
print("\n=== Test 2: Already consistent ===")
assert result2 == test2, "FAIL: Should be unchanged"
print("PASS")

# Test 3: Three headings, majority have ###
test3 = """### （1）A

内容A

（2）B

内容B

### （3）C"""
result3 = _normalize_sibling_headings(test3)
print("\n=== Test 3: Majority vote ===")
for line in result3.split('\n'):
    if '（' in line and '）' in line:
        print(repr(line))
assert '### （2）B' in result3, "FAIL: (2) should get ###"
print("PASS")

# Test 4: Non-consecutive → should NOT be grouped
test4 = """（1）A

### （5）B"""
result4 = _normalize_sibling_headings(test4)
print("\n=== Test 4: Non-consecutive ===")
assert '（1）A' in result4 and not result4.startswith('#'), "FAIL: should stay plain"
assert '### （5）B' in result4, "FAIL: should stay ###"
print("PASS")

# Test 5: Half-width parens
test5 = """(1) Section A

### (2) Section B"""
result5 = _normalize_sibling_headings(test5)
print("\n=== Test 5: Half-width parens ===")
for line in result5.split('\n'):
    if '(1)' in line or '(2)' in line:
        print(repr(line))
assert '### (1) Section A' in result5, "FAIL: half-width (1) should get ###"
print("PASS")

print("\n=== ALL TESTS PASSED ===")
