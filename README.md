# d2l_recurrence
To record my personal learning progress of this book.

## Function Parameters
```python
->help(torch.ones)

Help on built-in function ones in module torch:
ones(...)
    ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Returns a tensor filled with the scalar value `1`, with the shape defined

```

①	位置参数（Positional）	a, b

②	可变位置参数（*args）	*args

③	关键词参数分隔符（仅是 *）	*

④	关键字参数（Keyword-only）	c=1, d=2（必须写成 c=...）

⑤	可变关键字参数（**kwargs）	**kwargs

这个问题问得特别好，背后其实是 Python 语言设计的一次重要升级！我们来一起聊聊为啥 Python 3.9 以后可以用小写 `list`、`dict` 来做类型标注。

---

## 泛型标注

| Python 版本 | 类型标注写法 | 是否支持 |
|-------------|---------------|----------|
| 3.8 及以前 | `List[int]`（from typing） | ✅ 支持 |
| 3.9+        | `list[int]`   | ✅ 支持（推荐） |
| 3.8 及以前 | `list[int]`    | ❌ 报错 |
| 3.9+        | `List[int]`（from typing） | ✅ 仍然兼容，但不推荐 |

---

## 🔍 小细节补充：

1. `typing.List[int]` 是 **泛型类**
2. `list[int]` 是内置类 `list` 支持的语法糖（靠魔法方法实现）
3. Python 3.10 和 3.11 进一步支持了更多简洁类型标注，比如 `|` 运算符：

```python
def f(x: int | None) -> str | None:
    ...
```

等价于以前的：

```python
from typing import Optional

def f(x: Optional[int]) -> Optional[str]:
    ...
```
---
