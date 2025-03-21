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

在 Python 3.9 之前，内置类型（如 `list`、`dict`、`tuple`）**不支持下标操作**，也就是说：

```python
list[int]  ❌ 报错（在 3.8 及更早版本）
```

所以我们才需要引入 `typing.List` 这种**专门的类型标注版本**。

---

### 🎉 Python 3.9 改进了这一点：

> 内置的容器类型（`list`、`dict`、`tuple` 等）开始支持类型注解的“下标”语法！

也就是说，从 Python 3.9 开始，你可以这样写：

```python
def foo(data: list[int], mapping: dict[str, float]) -> tuple[int, float]:
    ...
```

这背后的实现是：Python 的内置类型实现了 **`__class_getitem__`** 魔法方法，使得 `list[int]` 成为合法的表达式，等效于以前的 `typing.List[int]`。

---

## ✅ 对比一下：

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

## 💡 总结

| 特性 | Python 3.8 及以下 | Python 3.9+ |
|------|-------------------|-------------|
| 类型标注 | `List[int]`, `Dict[str, float]` | `list[int]`, `dict[str, float]` |
| 写法简洁 | ❌ | ✅ |
| 推荐方式 | 使用 `typing` | 使用小写内建类型 |

---

如果你平时写代码时希望兼容旧版本，可以继续用 `typing.List`。但如果你确定 Python >= 3.9，完全可以用小写类型，简单、优雅、现代。

需要我帮你判断你的代码环境是不是 3.9+，或者搞一个向后兼容的写法示例也可以哈～