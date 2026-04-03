---
applyTo: "**"
---
# 禁止面向答案编程 — 通用性优先

## 铁律

**永远不要为某一类输入创建专用分支。** 发现某种输入效果差时，必须追问：

1. **根因是什么？** 不是 "PPT PDF 效果差"，是 "全页嵌入图干扰了合并逻辑 + 模型在薄文本层时跳过 OCR"
2. **这个根因是否只影响这一类输入？** 通常不是——全页嵌入图、薄文本层可以出现在任何 PDF 类型中
3. **修复能否让所有输入都受益？** 如果能，就是正确的修复

## 禁止的模式

```python
# ❌ 类型检测 + 专用分支
if _is_presentation_pdf(doc):
    prompt = SYSTEM_PROMPT_PRESENTATION
    # 专用逻辑...

# ❌ 为特定格式硬编码规则
if file_ext == ".xlsx":
    use_special_parser()

# ❌ 用 if/elif 链处理 N 种输入类型
if pdf_type == "presentation": ...
elif pdf_type == "textbook": ...
elif pdf_type == "form": ...
```

## 正确的做法

```python
# ✅ 数据驱动的通用机制
if image_area / page_area > 0.80:
    skip  # 任何 PDF 类型的全页图都会被过滤

# ✅ 基于实际数据而非类型标签做决策
if len(page_text) < 50:
    add_ocr_hint()  # 不管是 PPT、扫描件还是表单

# ✅ 提升管线的通用能力，而非打补丁
# 修改 prompt 让模型在文本层不完整时主动 OCR
# 而不是写一个 "PPT 专用 prompt"
```

## 自检清单

改代码前问自己：

- [ ] 我是在修通用管线，还是在给特定输入打补丁？
- [ ] 如果明天出现第 101 种 PDF 类型，我的修改还能用吗？
- [ ] 我的判断条件是基于数据特征（面积、文本长度）还是基于类型标签（is_presentation）？
- [ ] 相同的根因是否还影响其他场景？一次修好而不是重复修

## 一句话原则

> 100 种 PDF 不应该有 100 个 if 分支。找到共性根因，用一条通用规则解决。
