# Python 科目二 55 知识点

> 本文档整理自原始 docx 文档，涵盖 Python 面试/考试常见知识点与代码示例。

## 目录

- [1.1 关于 Python 关键知识点](#11-关于-python-关键知识点)
- [1.2 变量及其作用域](#12-变量及其作用域关键知识点)
- [值与引用，深浅拷贝](#值与引用深浅拷贝)
- [集合的基本操作](#集合的基本操作)
- [Bool](#bool)
- [浮点运算](#浮点运算)
- [两种除法](#两种除法)
- [异常处理](#异常处理)
- [包管理（import 及其加载机制）](#包管理import-及其加载机制)
- [生成器、迭代器与可迭代对象](#生成器yield迭代器与可迭代对象-iternext推导式)
- [面向对象编程](#面向对象编程类与对象继承重写静态方法)
- [继承](#继承)
- [多继承](#多继承一个子类拥有多个父类)
- [property](#property)
- [classmethod 和 staticmethod](#classmethod-和-staticmethod)
- [__eq__ 与 super().__init__](#__eq__-与-super__init__)
- [内嵌函数、返回函数与闭包](#内嵌函数返回函数与闭包)
- [匿名函数（lambda）](#匿名函数lambda)
- [装饰器](#装饰器)
- [并行与并发设计知识](#并行与并发设计知识线程与进程代码性能分析)
- [偏函数、高阶函数](#偏函数高阶函数partialmapreducefilter-等)
- [python 中的 and 与 or](#python-中的-and-与-or)
- [dir 与 dict](#dir-与-dict)
- [list append 和 extend](#list-append-和-extend)
- [Python 文件前面的顺序](#python-文件前面的顺序)
- [frozenset() 用法](#frozenset-用法)
- [空格](#空格)
- [空行](#空行)
- [大小写命名](#大小写命名)
- [注意1](#注意1)
- [字符串操作（格式化与拼接）](#字符串操作格式化与拼接)
- [注意2（安全）](#注意2安全)
- [路径名](#路径名)
- [序列化](#序列化)
- [通过代码覆盖分析进行测试补充](#通过代码覆盖分析进行测试补充)
- [逗号、括号与字符串（易错题）](#逗号括号与字符串易错题)
- [getrefcount](#getrefcount)
- [运算符的优先级](#运算符的优先级)
- [元组](#元组)
- [列表](#列表)
- [字典](#字典)
- [可变与不可变对象](#可变与不可变对象)
- [Decimal](#decimal)
- [CProfile](#cprofile)
- [Datatime](#datatime)
- [Random](#random)
- [Tracemalloc](#tracemalloc)
- [正则表达式](#正则表达式)
- [Unittest](#unittest)
- [numpy](#numpy)
- [PDB](#pdb)
- [Log](#log)
- [Type() 创建类](#type-创建类)
- [几类工具](#几类工具)
- [性能优化](#性能优化)
- [Python 链式比较](#python-链式比较)
- [疑问整理](#疑问整理)

---

## 1.1 关于 Python 关键知识点

1. 了解 Python 特点：解释型语言（vs 编译型语言）
2. 知道 Python 的执行过程原理（py -> pyc -> python 解释器）

## 1.2 变量及其作用域（关键知识点）

在 python 的函数内，可以直接引用外部变量，但不能改写外部变量。如果在函数内改写，等于在函数内也有一个同名变量。ID 会变的。

```python
i = 3
print(id(i))

def foo_1(x):
    def bar():
        i=i+4  #
        print(id(i))
        print(i)
```

- 情况一：`i=i+4`，报错 `local variable 'i' referenced before assignment`
- 情况二：若注释掉 `i=i+4`，则两个 ID 是一样的。

2. 可以使用 `nonlocal` 和 `global` 来实现。Nonlocal 与 global 的区别在于 nonlocal 语句会去搜寻本地变量与全局变量之间的变量，其会优先寻找层级关系与闭包作用域最近的外部变量。

```python
class Example(object):
    sum = 10

    def __init__(self):
        self.x = 11
        self.y = 12

    def add(self):
        return self.x + self.y
```

```python
example = Example()
print(example.add())    # 23

Example.sum = 20        # 修改了类的属性
example.sum = 13        # 动态创建了一个实例属性
print(Example.sum, example.sum)   # 20 13
```

4. 推导式 `[]`、`{}`、`()`，`lambda` 也是影响作用域，可以看做函数

```python
class A:
    a=2
    b=[a+i for i in range(3)]
    print(b)
# NameError: name 'a' is not defined
```

```python
class A:
    a=2
    b=[i for i in range(3)]   # 局部变量，外部用不了
    print(locals())

    for i in range(3):
        pass

    print(locals())
    print(b)
# {'__module__': '__main__', '__qualname__': 'A', 'a': 2, 'b': [0, 1, 2]}
# {'__module__': '__main__', '__qualname__': 'A', 'a': 2, 'b': [0, 1, 2], 'i': 2}
# [0, 1, 2]
```

```python
def func1():
    a=42
    b=[a+i for i in range(10)]
    print(b)

func1()
# [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# 正常输出
```

## 值与引用，深浅拷贝

关键知识点：

- 入参类型、入参默认值设定
- 值传递、引用传递
- 引用传递及操作对原变量的影响
- 深拷贝、浅拷贝

在 Python 中，对象赋值实际上是对对象的引用。

1. 当创建一个对象，然后把它赋值给另一个变量的时候，python 并没有拷贝这个对象，而只是拷贝了这个对象的引用。如果需要拷贝对象，需要使用标准库中的 copy 模块。copy 模块提供 `copy` 和 `deepcopy` 两个方法：

- **copy 浅拷贝**：拷贝一个对象，但是对象的属性还是引用原来的。对于可变类型，比如列表、字典、集合，只是复制其引用。基于引用所作的改变会影响到被引用对象。
- **deepcopy 深拷贝**：创建一个新的容器对象，包含原有对象元素（引用）全新拷贝的引用。外围和内部元素都拷贝对象本身，而不是引用。

Notes：对于数字，字符串和其他原子类型对象等，没有被拷贝的说法。如果对其重新赋值，也只是新创建一个对象，替换掉旧的而已。使用 copy 和 deepcopy 时，需要了解其使用场景，避免错误使用。

典型问题举例：可变对象与不可变对象的区别在于对象本身是否可变。

1. 可变对象：`list dict set`；不可变对象：`tuple string int float bool`。

```python
>>> a = [1, 2, 3]
>>> id(a)
2139167175368
>>> a[1] = 4
>>> id(a)
2139167175368
```

1. 可变对象的 id 一致，也就是指向同一个地址，所以修改 b 也修改 a。

```python
>>> a = (1, 2, 3)
>>> id(a)
2139167074776
>>> b = a              # b指向a
>>> a = (4, 5, 6)      # a变为另外一个对象了
>>> a
(4, 5, 6)
>>> b
(1, 2, 3)
>>> id(a)
2139167075928
>>> id(b)
2139167074776
```

1. 函数的参数默认值：只会在函数声明时初始化一次之后不会再变；

函数声明，只有参数个数相同时，才是同一个声明；所以那些传入参数的调用，不是同一个函数声明了；只有缺省第二个参数的，才是同一类调用。

```python
def extenList(val, List = []):
    print("[info][list is] ", List)
    List.append(val)
    return List

list1 = extenList(10)            # 1、7 List形参为空，是同一个调用，其实返回的list ID是一样的。
list2 = extenList(123, [])       # 2和3的实参虽然为空，但不是同一个调用。
list3 = extenList(12, [])
list4 = extenList("a")           #
list5 = extenList(14, [3])
list6 = extenList(194, [3])
list7 = extenList("b")           #

print(id(list1),id(list4),id(list7))
print(list1)
print(list2)
print(list3)
print(list4)
print(list5)
print(list6)
print(list7)
```

```text
------------------
[info][list is] []
[info][list is] []
[info][list is] []
[info][list is] [10]       # 4是同一个调用
[info][list is] [3]
[info][list is] [3]
[info][list is] [10, 'a']  # 7是同一个调用，所有 list，不断增加。
73214312 73214312 73214312
[10, 'a', 'b']
[123]
[12]
[10, 'a', 'b']
[3, 14]
[3, 194]
[10, 'a', 'b']
```

4. 在类里面的也一样

```python
class Example(object):
    def __init__(self, x_list = [1, 2, 3]):
        self.x_list = x_list
        print(id(x_list))
        print(id(self.__init__))

    def add(self, x):
        self.x_list.append(x)

    def sum(self):
        return sum(self.x_list)

example = Example()
print(example.sum())                # 6
example_1 = Example()
print(example_1.add(1))             # 不返回
print(example_1.sum())              # 7
print(example.sum())                # 仍然是7，由于在__init__(self, x_list = [1, 2, 3]):这几个函数指向同一个一个x_list = [1, 2, 3]，其ID是一样的。也说明，同一个类中的函数是指向同一个函数对象。
example_2 = Example()
print(example_2.sum())
```

```text
---------------
67847112
55130392
6
67847112
55130392
None
7
7
67847112
5513039
```

5. 在函数里面使用外部变量，注意。

```python
i = 2
print(id(i), i)

def f():
    i= i+3
    print(id(j), j)

f()
print(i)
# i= i+3 会报错，UnboundLocalError
# local variable 'i' referenced before assignment
```

这里的 i 与可变、不可变对象没有关系。必须添加 `global i`。

6. 切片，还有 `list()`、`type(A)`（`A`）也是浅拷贝。

```python
# 5.type(A)(A)是对A的深拷贝还是浅拷贝
>>> a = [[1,2,3],[4,5,6]]
>>> b = list(a)
>>> b[0][1] = 5
>>> b
[[1, 5, 3], [4, 5, 6]]
>>> a
[[1, 5, 3], [4, 5, 6]]
# 是浅拷贝。
```


## 集合的基本操作

> 原文档此处内容缺失，仅保留标题。

## Bool

`False, 0, '', [], {}, (), set()` 都可以视为 False，但不是 None。

- 空不可变对象 `is` 空不可变对象 为 True；
- 空可变对象 `==` 空可变对象 为 True
- `is` 比较的是 id 是否相同
- `==` 比较的是值是否相同

```python
a = 1
b = 1
print(id(a))   # 2643949349168
print(id(b))   # 2643949349168
print(a is b)  # True
```

空不可变对象的 id 都相同。比如 str，tuple：

```python
a = ''''
b = ''
print(id(a))   # 2678646785648
print(id(b))   # 2678646785648
print(a is b)  # True
```

```python
a = []
b = []
print(id(a))   # 空可变对象的ID是不一样的 2290017459072
print(id(b))   # 2290021949376
print(a is b)  # False
print(a == b)  # True
```

序列对象通常可以与相同序列类型的其他对象比较。这种比较使用字典式顺序。

## 浮点运算

原因是 1.125 是精确存储，而 1.115 不是精确存储。一个容易发现的原因就是十进制小数转二进制时精度丢失的问题，即 Python 存储 1.115 的时候实际上是 1.1149...，而对 1.1149 保留 2 位小数，当然是 1.11 而不是 1.12 了。

```python
from decimal import Decimal
print(Decimal(1.115))    # 1.1149999999999999911182158029987476766109466552734375
print(Decimal(1.125))    # 1.125
```

```python
from decimal import Decimal,getcontext
print('%.5f' % 3.14)  # 3.14000
print(Decimal('3.145')) # 3.145
getcontext().prec=8
print(Decimal(3.14))    # 3.140000000000000124344978758017532527446746826171875
print(Decimal(1)/Decimal(7))  # 0.14285714
print(Decimal(3))     # 3
```

```python
Decimal('1.135').quantize(Decimal('.01'))   # 1.13
Decimal('1.145').quantize(Decimal('.01'))   # 1.14
```

保留精度的 quantize 方法有一个默认参数 rounding，python 源码中可以看到：`quantize(self, exp, rounding=None, context=None)`，可追溯到 rounding 有一个默认值 `rounding=ROUND_HALF_EVEN`。

**ROUND_UP**：舍弃小数部分非 0 时，在前面增加数字，如 5.21 -> 5.3；-基础的

```python
print(Decimal(3.141).quantize(Decimal('.0001'),rounding=ROUND_UP))   # 3.1411
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_UP))     # 3.15
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_UP))    # -3.15
```

**ROUND_DOWN**：舍弃小数部分，从不在前面数字做增加操作，如 5.21 -> 5.2；-基础

UP 和 DOWN 正负正常添加，不考虑正负。舍入方向为 0。

**ROUND_CEILING**：如果 Decimal 为正，则做 ROUND_UP 操作；如果 Decimal 为负，则做 ROUND_DOWN 操作；--- 舍入方向到无穷大。

```python
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_CEILING)   # 3.15
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_CEILING)) # -3.14
```

**ROUND_FLOOR**：如果 Decimal 为负，则做 ROUND_UP 操作；如果 Decimal 为正，则做 ROUND_DOWN 操作；--- 舍入方向到无穷小

```python
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_FLOOR))   # -3.15
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_FLOOR))    # 3.14
```

CEILING 和 FLOOR，负数刚好反过来【注意正负】，整数 CEILING=UP，DOWN=FLOOR。

**ROUND_HALF_DOWN**【标准的五舍六入，中间向下】：如果舍弃部分 > .5，则做 ROUND_UP 操作；否则，做 ROUND_DOWN 操作；

```python
print(Decimal('3.135').quantize(Decimal('.01'),rounding=ROUND_HALF_DOWN))  # 3.13
Decimal('-3.136').quantize(Decimal('.01'),rounding=ROUND_HALF_DOWN)         # -3.14
```

**ROUND_HALF_UP**【标准的四舍五入，中间向上】：如果舍弃部分 >= .5，则做 ROUND_UP 操作；否则，做 ROUND_DOWN 操作；不考虑正负。

**ROUND_HALF_EVEN**：四舍五入，5 特别，如果最后一位是 5，则检查前一位，奇数在向上，偶数则向下。

```python
print(Decimal('3.135').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.14; 到时第二位是奇数，则上一位，变为3.14
print(Decimal('3.145').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.14; 到时第二位是偶数，向下，直接保留3.14
print(Decimal('3.146').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.15; 直接五入，变为3.15
```

注意 `Decimal('3.146')`，`Decimal(3.146)` 的差别较大，要注意用字符串。

## 两种除法

`//` 增加了一个操作符 `//`，以执行地板除：`//` 除法不管操作数为何种数值类型，总是会舍去小数部分，返回数字序列中比真正的商小的最接近的数字。

```python
print(1/2)   # 0.5
print(1.0/2.0)   # 0.5
print(7//2)   # 3
print(-7//2)  # -4, 类似ROUND_FLOOR
```


## 异常处理

捕获父类异常的情况下，抛出子类的信息：

```python
class ParentException(Exception):
    def __init__(self):
        Exception.__init__(self, "Parent Exception")

class ChildException(ParentException):
    def __init__(self):
        Exception.__init__(self, "Child Exception")

try:
    raise ChildException
except ParentException as exc:
    print("catch exception: %s" % exc)
# catch exception: Child Exception
```

为什么？因为异常是子类，子类也是父类的一种，所以可以被父类捕获。

**Finally 里面的返回最大：**

```python
def divide(x, y):
    try:
        return x/y
    except Exception:
        return 0
    finally:
        return -1

print(divide(1, 0))
print(divide(1, 1))
#
-1
-1
```

1. 使用 try…except… 结构对代码作保护时，需要在异常后使用 finally…结构保证操作对象的释放
2. 不要使用 `except:` 语句来捕获所有异常，这样对于 bug 定位不好，except 后面一定要有异常的点。
3. 避免 finally 中可能发生的陷阱，不要在 finally 中使用 return 或 break 语句，finally 一定会执行
4. 不在 except 分支里面的 raise 都必须带异常类型，在分支里面可不带。【学习：try，except，finally】
5. 尽量用异常来表示特殊情况，而不要返回 None。

```python
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        raise ValueError('Invalid inputs') from e
```

```python
class AuthenictionError(Exception):
    def __init__(self):
        super(AuthenictionError, self).__init__()
```

```python
class AuthenictionError(BaseException):
    def __init__(self):
        super(AuthenictionError, self).__init__()
```

不要使用 BaseException 的派生：+-- 系统出口、-键盘中断、--发电退出，+-- 例外情况。这三个不例外的事实上并不是错误，这意味着通常你不想把它们当作错误来捕捉。

## 包管理（import 及其加载机制）

Python 导包的规范写法、规则、模块的封装

- `__all__ = [xxxx]`
- 包冲突
- 包冲突处理
- 动态导包，会应用
- import 机制与 `__init__.py`

python 执行 import 时进行如下操作：

1. 第一步，创建一个新的，空的 module 对象（它可能包含多个 module）。如果需要导入的 module 的名字是 m1，则解释器必须找到 m1.py；
   首先在当前目录查找，然后是在环境变量里 PYTHONPATH 中查找。如果当前路径或 PYTHONPATH 中存在与标准 module 同样的，如果当前目录下存在 xml.py，那么执行 import xml 时，导入的是当前目录下的 module，而不是系统标准的 xml。
2. 第 二步，把这个 module 对象插入 sys.module 中。
3. 第 三 步，装载 module 的代码（如果需要，首先必须编译）。
4. 第 四 步，执行新的 module 中对应的代码。

python 中的 package 必须包含一个 `__init__.py` 的文件，可以为空，只要它存在就说明该目录该被看作一个 package 处理。

python3 中的包是否不需要 `__init__.py`？python3.3+ 中支持隐式名称空间包，允许创建不带 init 的包（但是只适用于空的 init），也就是说在 python3.3 后，空的 init 文件不再是必须。

`__all__` 变量是一个由 string 元素组成的 list 变量。写在 `__init__.py` 内。它定义了当我们使用 `from import *` 导入某个模块的时候能导出的符号（这里代表变量，函数， 类等）。`from import *` 默认的行为是从给定的命名空间导出所有的符号（当然下划线开头的变量，方法和类除外）。需要注意的是 `__all__` 只影响到了 `from import *` 这种导入方式，对于 `from import <member>` 导入方式并没有影响，仍然可以从外部导入。

导入的包中含有相同的函数名：会导致覆盖，后面导入的覆盖前面的。

## 生成器（yield）、迭代器与可迭代对象（iter/next）、推导式

1）可迭代对象包含迭代器。

2）如果一个对象拥有 `__iter__` 方法，其是可迭代对象；如果一个对象拥有 `next` 方法，其是迭代器。

3）定义可迭代对象，必须实现 `__iter` 方法；定义迭代器，必须实现 `__iter` 和 `next` 方法。

**如何实现一个对象可迭代**

- 第一种，实现 `__iter__`，通过 yield 生成迭代器
- 第二种，实现 `__iter__`，返回自身，实现 `__next__`
- 第三种，实现 `__getitems__`，接收角标，返回元素

生成器 yield 是一种特殊的迭代器，生成器自动实现了“迭代器协议”（即 iter 和 next 方法），不需要再手动实现两方法。yield 可以理解为 return，返回后面的值给调用者。不同的是 return 返回后， 函数会释放，而生成器则不会。在直接调用 next 方法或用 for 语句进行下一次迭代时，生成器会从 yield 下一句开始执行，直至遇到下一个 yield。

```python
class Next(object):
    def __init__(self):
        self.data = [0, 1, 2, 3, 4]
        self._inter = iter(self.data)
    def getLen(self):
        return len(self.data)
    def __iter__(self):
        return self
    def __call__(self):
        return next(self._inter)
    def __next__(self):
        return next(self._inter)
    # 括号里面是多少，就输出多少个，如果括号中的数字大于迭代器中的数字，则输出迭代器中全部内容，不会报错
for it in iter(Next(), 3):
    print(it)
# 0
# 1
# 2
```


## 面向对象编程（类与对象、继承、重写、静态方法）

关键知识点：

- 各种继承后的变量的值
- 父类方法调用、子类方法调用
- 类方法、静态方法、实例方法

知识点：对于所有的类都有以下属性，用 `dir()` 可以查看：

- `_ _ name_ _`：类的名字（字符串）
- `_ _ doc _ _ `：类的文档字符串
- `_ _ bases _ _`：类的所有父类组成的元组
- `_ _ dict _ _`：类的属性组成的字典
- `_ _ module _ _`：类所属的模块
- `_ _ class _ _`：类对象的类型

在 Python 中，以单下划线开头表示的是 protected 类型的变量，即只能允许其本身与子类进行访问。一般约定以单下划线开头的函数为模块私有的，也就是说 `from moduleName import *` 将不会引入以单下划线"`_`"开头的函数。

双下划线的表示的是 private 类型的变量。只能是允许这个类本身进行访问了，连子类也不可以。这类属性在运行时属性名会加上单下划线和类名。单下划线，可被重写，调用子类方法；双下划线，不能被重写，调用的还是父类方法。两种方法无法真正避免不被外界调用。

```python
class A:
    def __init__(self):
        self.__j=1
        self.number=5

class B(A):
    def __init__(self):
        self.__j=2
        self.number=7
    def show(self):
        print(self.__j,self.number)

b=B()
print(b.__dir__())
#print(b.__j)  # 'B' object has no attribute '__j'
print(b._B__j)  # 2
print(b.__dir__())
# ['_B__j', 'number', '__module__', '__init__', 'show', '__doc__', '__dict__', '__weakref__', ...]
```

对于简单的类公有变量，最好不要设置过多的 getter/setter，直接访问即可。

**del 与 Python 垃圾回收机制**

```python
class Dog:
    def __del__(self):   # 当内存不需要的时候调用这个删除方法，python解释器自动调用
        print("英雄over")

dog1=Dog()   # 创建一个对象
dog2=dog1
del dog1     # 因为dog2还指向Dog1的地址，该地址的计数还未归零，python的垃圾回收机制没有被
print("==========")
del dog2     # 计数归零，触发垃圾回收机制，执行__del__
print("==========")
# ==========
# 英雄over
# ==========
```

```python
class Dog:
    def __del__(self):
        print("英雄over")

dog1=Dog()
dog2=dog1
del dog1
print("==========")
print("==========")
# ==========
# ==========
# 英雄over
```

与上一个代码块不同之处在于垃圾回收机制是在程序执行之后、释放内存，由 python 执行，所以英雄 over 在最后。

**方法调用的覆盖（没有双下划线时）：**

```python
class A:
    def dis(self):
        return self.dis1()   # 没有双下划线,先在自己的实例里面看，有无这个函数
    def dis1(self):
        self.a=12
        print(self.a,"A")

class B(A):
    def dis1(self):
        self.a=13
        print(self.a,"B")

print("zhixing")
a=A()
b=B()
b.dis()
a.dis()
# Zhixing
# 13 B
# 12 A
# b会调用自己的方法，a也会调用自己的方法，是A有特殊情况在A类里面，可以掉子类的方法
```

**方法调用的覆盖（有双下划线时）：**

```python
class A():
    def __check(self):
        print("A")
    def display(self):
        self.__check()  # 有双下划线，实际调用self._A__check()

class B(A):
    def __check(self):
        print("B")

a = A()
b = B()
a.display()  # A
b.display()  # A
```


## 继承

如果子类没有定义自己的初始化函数，父类的初始化函数会被默认调用；

但是如果要实例化子类的对象，则只能传入父类的初始化函数对应的参数，否则会出错。

```python
# self.number=5
class B(A):
    def __init__(self):
        self.__j=2       # 这个部分，双下划线表示的父类的属性，如果子类的init方法中，没有显示调用父类的初始化函数，那么父类的属性不会被初始化，所以下面的b不会有_j这个属性
        self.number=7
    def show(self):
        print(self.__j,self.number)

b=B()
print(b.__dir__())
print(b.__j)
# 'B' object has no attribute '__j'
print(b._B__j)
# 2 ----------- 返回'_B__j' __j
```

如果子类定义了自己的初始化函数，在子类中显示调用父类，子类和父类的属性都会被初始化。super 主要用来调用父类方法来显示调用父类，在子类中，一般会定义与父类相同的属性（数据属性，方法），从而来实现子类特有的行为。也就是说，子类会继承父类的所有的属性和方法，子类也可以覆盖父类同名的属性和方法。

- `__new__`：第一个参数是 cls
- `__init__`：第一个参数是 self

若 `__new__()` 没有正确返回当前类 cls 的实例，那么 `__init__` 将不会被调用：

```python
class A:
    def __new__(cls, *args, **kwargs):
        print("A' __new__")
        return object.__new__(cls)   # 是关键
    def __init__(self):
        print("A' __init__")

class B(A):
    def __new__(cls, *args, **kwargs):
        print("B' __new__")
        # return object.__new__(cls)   # new必须要有返回值。
    def __init__(self):
        self.bb=0
        print("B' __init__")

b=B()   # B' __new__ 没有创建真正的类B，所以类B的构造函数没有调用
a=A()
print(type(a))   # <class '__main__.A'>
print(type(b))   # <class 'NoneType'>    b实例没有东西
b.bb=20   # 报错 'NoneType' object has no attribute 'bb'
if not b:
    print("None")   # 会打印出来
```

## 多继承：一个子类拥有多个父类

若父类中有相同的属性时：

如果两个父类有相同的属性时，前面父类的属性会有效，后面的不生效。这与模块导入完全不一样。

```python
class A:
    def test(self):
        print("--A--打印test")

class B:
    def test(self):
        print("--B--打印test")

class C(B,A):
    pass

c = C()
c.test()
print(C.__bases__)
# print("--B--打印test")
# (<class '__main__.B'>, <class '__main__.A'>)，不会再打印，<class 'object'>
```

如果多个父类直接也有继承关系，为提高查找的性能，会采用深度优先的算法去父类中找方法或者属性，如果找过的父类，其他类不会再找一遍。比如：

D，集承 B 和 C，会优先 B，A；然后 C。C 不会再找 A。

如果想在重写了父类的方法之后还想调用父类的方法，那么我们就可以使用 `super().方法名()` 来在重写父类方法的基础上来增加自己的代码。

```python
class Animal():
    def sleep(self):
        print("睡觉")
    def brak(self):
        print("动物叫")

class Dog(Animal):
    def bark(self):
        # 调用父类的brak()
        super().brak()
        print("狗叫")

dog = Dog()
dog.bark()
```

```python
class TestClass(object):
    # 类变量
    val1 = 100
    def __init__(self):
        # 成员变量
        self.val2 = 200
    def fcn(self, val=400):
        val3 = 300   # 这个是fcn的局部变量，不是类变量，也不是实例变量
        self.val4 = val
        self.val5 = 500

inst1 = TestClass()
print(dir(inst1))
# 这里面有 val1和fcn等，
inst2 = TestClass()
print(TestClass.val1) # 100，类变量
print(inst1.val1)     # 100，直接使用类变量
inst1.val1 = 1000     # 增加一个自己的实例变量，两个 val1 分开
print(inst1.val1)     # 1000
print(TestClass.val1) # 100
print(inst2.val1)     # 100
TestClass.val1 = 2000
print(inst2.val1)     # 2000 没有重新赋值，跟类变量保存一致
print(TestClass.val1) # 2000
print(inst1.val1)     # 1000 被重新赋值后，就跟类变量没关系了
inst3 = TestClass()
```


## property

由于 python 进行属性的定义时，没办法设置私有属性，因此要通过 @property 的方法来进行设置。这样可以隐藏属性名，让用户进行使用的时候无法随意修改。

```python
class TestClass(object):
    def __init__(self):
        self._val2 = 200
        self._val1 = 100

    @property
    def val22(self):
        return self._val2

    @property
    def val11(self):
        return self._val1

    @val11.setter   # 注意此点
    def val11(self,newvalue):
        self._val1=newvalue

a=TestClass()
a.val11=3000
print(a.val11)
```

## classmethod 和 staticmethod

一般的类方法要接收一个 self 参数表示此类的实例，但有些方法不需要访问实例，这时分为两种情况：

1. 方法不需要访问任何成员，或者只需要显示访问这个类自己的成员。这样的方法不需要额外参数，应当用 @staticmethod 装饰。
2. 方法不需要访问实例的成员，但需要访问基类或派生类的成员。这时应当用 @classmethod 装饰。装饰后的方法，其第一个参数不再传入实例，而是传入调用者的最底层类。

```python
class A(object):
    def m1(self, n):   # 实例方法
        print("self:", self)

    @classmethod      # 类方法
    def m2(cls, n):
        print("cls:", cls)

    @staticmethod     # 静态方法
    def m3(n):
        pass

a = A()
a.m1(1)   # self: <__main__.A object at 0x...>
A.m2(1)   # cls: <class '__main__.A'>
A.m3(1)
```

在类中一共定义了 3 个方法，m1 是实例方法，第一个参数必须是 self（约定俗成的）。m2 是类方法，第一个参数必须是 cls（同样是约定俗成），m3 是静态方法，参数根据业务需求定，可有可无。当程序运行时，大概发生了这么几件事（结合下面的图来看）。

- 第一步：代码从第一行开始执行 class 命令，此时会创建一个类 A 对象（没错，类也是对象，一切皆对象嘛）同时初始化类里面的属性和方法，记住，此刻实例对象还没创建出来。
- 第二、三步：接着执行 a=A()，系统自动调用类的构造器，构造出实例对象 a
- 接着调用 a.m1(1)，m1 是实例方法，内部会自动把实例对象传递给 self 参数进行绑定，也就是说， self 和 a 指向的都是同一个实例对象。
- 调用A.m2(1)时，python 内部隐式地把类对象传递给 cls 参数，cls 和 A 都指向类对象。

```python
print(A.m1)
print(a.m1)
#
print(A.m2)
print(a.m2)
#
print(A.m3)
print(a.m3)
# <function A.m1 at 0x03F72390>
```

- `A.m1` 是一个还没有绑定实例对象的方法，对于未绑定方法，调用 A.m1 时必须显式地传入一个实例对象进去。`A.m1(a, 1)` 等价 `a.m1(1)`
- `a.m1` 是已经绑定了实例的方法，python 隐式地把对象传递给了 self 参数，所以不再手动传递参数，这是调用实例方法的过程。
- 类方法，绑定在类上，`A.m2(1)` 等价 `a.m2(1)`
- 类方法，绑定在类上
- `m3` 是类里面的一个静态方法，跟普通函数没什么区别，与类和实例都没有所谓的绑定关系，它只不过是碰巧存在于类中的一个函数而已。不论是通类还是实例都可以引用该方法。

**classmethod 对每个类做独立计数：**

```python
class Spam:
    num_instances = 0      # 类变量
    @classmethod
    def count(cls):        # 对每个类做独立计数
        cls.num_instances += 1   # cls是实例所属的最底层类
    def __init__(self):
        self.count()       # 把self.__class__传给count方法

class Sub(Spam):
    num_instances = 0

class Other(Spam):
    num_instances = 0

x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()

print(x.num_instances, y1.num_instances, z1.num_instances)   # 输出：(1, 2, 3)
print(Spam.num_instances, Sub.num_instances, Other.num_instances)   # 输出：(1, 2, 3)
```

```python
class Spam:
    num_instances = 0
    @classmethod
    def count(cls):        # 对每个类做独立计数
        cls.num_instances += 1   # cls是实例所属的最底层类，当底层类没有时，使用父类的变量
        print(cls, cls.num_instances)
    def __init__(self):
        self.count()       # 把self.__class__传给count方法

class Sub(Spam):
    pass

class Other(Spam):
    pass

x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()
print(x.num_instances, y1.num_instances, z1.num_instances)   # 输出：(1, 3, 4)
print(Spam.num_instances, Sub.num_instances, Other.num_instances)   # 输出：(1, 3, 4)
# <class '__main__.Spam'> 1
# <class '__main__.Sub'> 2
# <class '__main__.Sub'> 3
# <class '__main__.Other'> 2
# <class '__main__.Other'> 3
# <class '__main__.Other'> 4
```

**对基类基数（staticmethod）：**

```python
class Spam:
    num_instances = 0
    @staticmethod
    def count():         # 静态方法
        Spam.num_instances += 1   # 只是对Spam基类基数
    def __init__(self):
        count()          # Spam.count

class Sub(Spam):
    num_instances = 0    # 类变量

class Other(Spam):
    num_instances = 0    # 类变量

x = Spam()
y1, y2 = Sub(), Sub()
z1, z2, z3 = Other(), Other(), Other()
print(x.num_instances, y1.num_instances, z1.num_instances)   # 输出：(6, 0, 0)
print(Spam.num_instances, Sub.num_instances, Other.num_instances)   # 输出：(6, 0, 0)
```


## __eq__ 与 super().__init__

`__eq__` 是针对 `==` 比较的魔法方法，对于 `is` 就无能为力了。

```python
class AA:
    def __init__(self, name):
        self.name = name
    def __eq__(self, other):
        return self is other

class BB:
    def __init__(self, name):
        self.name = name
    def __eq__(self, other):
        print(str(self),str(other))
        return str(self) == str(other)

a = AA("abc")
b = BB("abc")
print(a == b)
print("aaa")
print(b == a)
# False
# aaa
# <__main__.BB object at 0x0431A890> <__main__.AA object at 0x0431A810>
# False
```

```python
class A:
    def __init__(self):
        self.__i = 1
        self.j = 5

class B(A):
    def __init__(self):
        self.__i = 2
        self.j = 7
        super().__init__()   # super的位置很关键
    def display(self):
        print(self.__i)
        print(self.j)

c=B()
c.display()
# 2
# 5
```

```python
class A:
    def __init__(self):
        print(self)

class B(A):
    def __init__(self):
        print(self)
        super().__init__()

c = B()
print(c)
# 都是指向同一个对象c。
# <__main__.B object at 0x044C9810>
# <__main__.B object at 0x044C9810>
# <__main__.B object at 0x044C9810>
```

## 内嵌函数、返回函数与闭包

```python
def func(name):
    def inner_func(age):
        print('name:', name, 'age:', age)
    return inner_func

f = func('name')
f(26)
```

在 python 的函数内，可以直接引用外部变量，但不能改写外部变量，因此如果在闭包中直接改写父函数的变量，就会发生错误。python 3 中引入了 nonlocal 语句解决了这个问题。

```python
def cnt(name):
    count = 0
    def counter():
        nonlocal count   # 注释这行报错：local variable 'count' referenced before assignment
        count += 1
        print(name, count)
    return counter

f = cnt('counter')
f()
```

闭包的最大特点是可以将父函数的变量与内部函数绑定，并返回绑定变量后的函数（也即闭包），此时即便生成闭包的环境（父函数）已经释放，闭包仍然存在，这个过程很像类（父函数）生成实例（闭包），不同的是父函数只在调用时执行，执行完毕后其环境就会释放，而类则在文件执行时创建，一般程序执行完毕后作用域才释放，因此对一些需要重用的功能且不足以定义为类的行为，使用闭包会比使用类占用更少的资源，且更轻巧灵活。

## 匿名函数（lambda）

lambda 函数作为闭包：

```python
def multipliers():
    return [lambda x: i * x for i in range(4)]
print([m(2) for m in multipliers()])
# [6,6,6,6]
```

```python
def multipliers():
    # 添加了一个默认参数i=i
    return [lambda x, i=i: i * x for i in range(4)]
print([m(2) for m in multipliers()])
# [0,2,4,6]
```

## 装饰器

1. python 解释器发现 @dobi，就去调用与其对应的函数（dobi 函数）
   - dobi 函数调用前要指定一个参数，传入的就是 @dobi 下面修饰的函数，也就是 g()
   - dobi() 函数执行打印 "function f"，调用 g()，g() 打印 "function g"

```python
def dobi(f):
    print("function f")   #定义过程
    return f()   #在定义过程中

@dobi
def g():
    print("function g")
# 注意，没有调用g（）,也会打印，此时是加载的过程
# function f
# function g
```

说明：

```python
def dobi(f):
    print("function f")
    return f()

@dobi
def g():
    print("function g")

g()   # 不能有调用
# NoneType' object is not callable
# File "D:\pythonPorject\1.py", line 10, in <module>
```

事实上，装饰器就是一种闭包的应用，只不过其传递的是函数。

```python
def makebold(f):
    def wrapped():
        return "<b>"+f()+"</b>"
    return wrapped
def makeitalic(f):
    def wrapped():
        return "<i>"+f()+"</i>"
    return wrapped

@makebold
@makeitalic
def hello():   # 等价于hello = makebold(makeitalic(hello))
    return "hello"

print(hello())
# <b><i>hello</i></b>
```

这里要有 print(hello())，否则啥也不打印。是指执行状态。先在 hello 的外面执行第一层装饰 @makeitalic，变为：`<i>hello</i>`；然后执行第二层装饰，变为：`<b><i>hello</i></b>`。

一共四个位置，两个在加载阶段（按照 21,12 进行加载，先加载 1，在加载 2），两个在执行/装饰阶段（先把 2 的前后执行完，然后在包一层 1 的前后，把 func 夹在里面，看前面的例子）。多个装饰器装饰一个函数时，执行时的顺序是：最先装饰的装饰器，最后一个执行。它遵循了先进后出规则，类似于 stack。

```python
def set_fun1(func):
    print("set_fun1已被定义")   # 打印用于验证在多个装饰器的情况下，多个装饰器之间的执行顺序
    def call_fun1(*args, **kwargs):
        print("call_fun1执行了")   # 当被装饰函数执行时，会打印
        return func()
    return call_fun1

def set_fun2(func):
    print("set_fun2已被定义")
    def call_fun2(*args, **kwargs):
        print("call_fun2执行了")
        return func()
    return call_fun2

# 装饰函数
@set_fun2
@set_fun1
def test():
    print("******恭喜你找到组织了******")

print("未执行")
test()
print("执行")
# set_fun1已被定义
# set_fun2已被定义
# 未执行     #以上部分同案例1，在调用test()前就会加载，定义
# call_fun2执行了
# call_fun1执行了
# ******恭喜你找到组织了******
# 执行
```

```python
def set_fun1(func):
    print("set_fun1已被定义")
    def call_fun1(*args, **kwargs):   #执行部分
        print("call_fun1执行了")
        return func()
    print("set_fun1定义结束")
    return call_fun1

def set_fun2(func):
    print("set_fun2已被定义")
    def call_fun2(*args, **kwargs):
        print("call_fun2执行了")
        return func()
    print("set_fun2定义结束")
    return call_fun2

@set_fun2
@set_fun1
def test():
    print("******恭喜你找到组织了******")

print("未执行")
test()
print("执行")
# set_fun1已被定义
# set_fun1定义结束
# set_fun2已被定义
# set_fun2定义结束
# 未执行
# call_fun2执行了
# call_fun1执行了
# ******恭喜你找到组织了******
```

执行时，把函数包在里面的情况：

```python
def deco_a(func):
    def wrapper(*args, **kwargs):
        print('deco_a')
        func(*args, **kwargs)
        print('deco_a1')
    return wrapper

def deco_b(func):
    def wrapper(*args, **kwargs):
        print('deco_b')
        func(*args, **kwargs)
        print('deco_b1')
    return wrapper

@deco_a
@deco_b
def test():
    print('function running')

print("dingyi")
test()
# dingyi  #这个是定义：
# deco_a
# deco_b， 第一层包装
# function running
# deco_b1， 第一层包装
# deco_a1
```

> 原文档练习：下列有关 python 中的闭包，错误的是（）
> - A. 某个函数被当成对象返回时，夹带了外部变量，就形成了一个闭包
> - B. 闭包常用在装饰器函数中，装饰器需要自定义参数时，一般都会形成闭包
> - C. 闭包中是可以修改【-读取-】外部作用域中除 nonlocal 之外的局部变量
> - D. 全部说法都不正确
>
> 答案：C

```python
func_list = []
for i in range(3):
    def closure(i):
        def my_func(a):
            return i + a
        return my_func
    func_list.append(closure(i))

for f in func_list:
    print(f(1))   # 1,2,3
```

打印，采用闭包的方式：必须要有函数的嵌套，而且外层函数必须返回内层函数。内层函数可以有返回值，也可以没有返回值。内层函数一定要用到外层函数中定义的变量，如果只满足了上一条也不算是闭包，一定要用到外层"包装函数"的变量，这些变量称之为"自由变量"。

对比（非闭包）：

```python
func_list = []
for i in range(3):
    def my_func(a):
        return i + a
    func_list.append(my_func)   # 定义三个函数，将三个函数放在一个列表中

for f in func_list:   # 调用列表中的三个函数
    print(f(1))   # 3,3,3
```


## 并行与并发设计知识（线程与进程、代码性能分析）

Python 相关的并发原理（基于 GIL），能分析出说法的正确与否：常用的线程、进程库：multiprocess、threading 等；线程的启停处理：理解 stop、run、join 等函数的作用，阻塞 or 非阻塞。

1. 全局解释器锁 GIL。全局解释器锁（GIL）只允许一次执行一个线程。Python 字节码解释器只有在一个机器指令完成后，另一个机器指令没开始前，才会进行线程切换。

多线程编程技术，要不弄个计数器，每个线程数到 100 就释放。对于 CPU 无效，但对于 I/O 还是有效的。不再用计数的方式，改用时间片的方式：每个线程的执行时间片是 5000 微秒。为了保证释放 GIL 后，不被自己马上又抢到，新增了一个锁实现强制线程切换，Python 的多线程适用于阻塞式 IO 的场景，不适用于并行计算的场景。使用协程来处理并发场景。单核多线程，多核多进程。

好处是线程安全的，同一时刻只有一个线程执行。

但长期以来，Python 最为人诟病的就是它有一把锁：GIL，这把锁让 Python 无法真正地实现多线程执行，无法利用多核 CPU 的高性能。实际上，这个锁跟 Python 没有半毛钱的关系，而是负责解释执行 Python 的解释器：CPython 的锅。CPython 是用 C 语言编写的 Python 解释器，也是最广为使用的 Python 解释器，一般在没有特殊说明时，说 Python 指的就是这个 CPython 解释器。JPython 没有 GIL，MultiProcess Python 提供了 MultiProcess，通过多进程的方式绕过 GIL。

**进程**：进程是系统进行分配资源和调度的基本单位，是操作系统执行的基本单元；每个进程都拥有唯一的地址空间：程序代码、堆、空闲、栈。进程是轮流使用 CPU 的，CPU 被若干进程共享，使用某种调度算法来决定何时停止一个进程，并转而为另一个进程提供服务。每一个进程独占一个 CPU 核心资源，在处理 I/O 请求的时候，CPU 处于阻塞状态。

```python
import multiprocessing
import threading
import time

n = 0
def count(num):
    global n
    for i in range(4):
        n += i
        print("Process {0}:n={1},id(n)={2}".format(num, n, id(n)))

if __name__ == '__main__':
    start_time = time.time()
    process = list()
    for i in range(2):
        p = multiprocessing.Process(target=count, args=(i,))   # 测试多进程使用
        # p = threading.Thread(target=count, args=(i,))   # 测试多线程使用
        p.start()
        p.join()

    print("Main:n={0},id(n)={1}".format(n, id(n)))
    end_time = time.time()
    print("Total time:{0}".format(end_time - start_time))
# Process 0:n=0,id(n)=2008343296
# Process 0:n=1,id(n)=2008343312
# Process 0:n=3,id(n)=2008343344
# Process 0:n=6,id(n)=2008343392
# Process 1:n=0,id(n)=2008343296
# Process 1:n=1,id(n)=2008343312
# Process 1:n=3,id(n)=2008343344
# Process 1:n=6,id(n)=2008343392
# Main:n=0,id(n)=2008343296
# Total time:4.221319675445557
```

**线程**：线程是 CPU 执行的最基本单元，多线程无需重复申请资源，子线程和父线程共享资源；多线程间的通信速度快于进程通信，效率更高。

一个进程可以有一个或多个线程，同一进程中的多个线程将共享该进程中的全部系统资源，如虚拟地址空间，文件描述符和信号处理等等。但同一进程中的多个线程有各自的调用栈和线程本地存储。

由于线程之间能够共享地址空间，因此，需要考虑同步和互斥操作；进程则不需要，全部虚拟空间。一个线程的意外终止会影响整个进程的正常运行，但是一个进程的意外终止不会影响其他的进程的运行。因此，多进程程序安全性更高。

**协程**：有一个线程在执行，只有当子程序内部发生阻塞或者 IO 时，才会交出线程执行权给其他子程序，适当的时候再返回；与多线程相比，协程的优势：（1）省去了大量线程切换的开销；（2）由于是单线程执行，共享资源不需要加锁，执行效率更高；当然，为了充分利用多核，推荐多进程+协程。

- 多线程推荐的库：threading；多进程推荐的库：multiprocessing

## 偏函数、高阶函数（partial、map、reduce、filter 等）

1. `map()` 函数接收两个参数，一个是函数，一个是 Iterable，map 将传入的函数依次作用到序列的每个元素，并把结果作为新的 Iterator 返回。

```python
def f(x):
    return x * x
r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])   #返回一个迭代器
print(list(r),type(r))
# [1, 4, 9, 16, 25, 36, 49, 64, 81] <class 'map'>
# 这里的F只带一个参数，多了不行
```

2. `reduce` 归纳，把一个函数作用在一个序列 [x1, x2, x3, ...] 上，这个函数必须接收两个参数，reduce 把结果继续和序列的下一个元素做累积计算。

```python
from functools import reduce

def f(x, y):
    return x + y

r = reduce(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(r, type(r))
# 45 <class 'int'>
# 这里的F只带两个参数，其他参数量不对会报错。
```

3. `filter` 只保留为 True 的，过滤为 False 的

```python
def f(x):
    if x % 2 == 0:
        return True
    else:
        return False
r = filter(f, [1, 2, 3, 4, 5])   # [2,4]
print(list(r), type(r))
# 这里的f只带一个参数，如果多了，少了会报错
```

4. `partial` 偏函数：固定原有函数部分参数，使调用更简单

```python
import functools
int2 = functools.partial(int,base=2)
print(int2('1000000'))
# 等价于
def int2(x, base=2):
    return int(x, base)
# 64
```

## python 中的 and 与 or

```python
a = {1, 2, 3} and {4, 5, 6}
b = {1, 2, 3} or {4, 5, 6}
print(a)   # {4, 5, 6}
print(b)   # {1, 2, 3}
# and计算所有表达式的值为真，那么就返回最后一个真值
# or计算所有的表达式的值为真，那么就返回第一个真值
```

## dir 与 dict

`__dict__` 是 `dir()` 的子集，`dir()` 包括父类属性。实例的 `__dict__` 仅存储与该实例相关的实例属性。`dir()` 函数会自动寻找一个对象的所有属性（包括从父类中继承的属性）。

## list append 和 extend

```python
ls=['2020','20.20','python']
ls.append(2020)
ls.append([2020,'2020'])
ls.extend([2020,'1111'])
print(ls)
# ['2020', '20.20', 'python', 2020, [2020, '2020'], 2020, '1111']
```

## Python 文件前面的顺序

```python
#!/usr/bin/env python
# coding: utf-8
"""【没有空格】版权所有 (c) 华为技术有限公司 2012-2020 这是一个模块文档字符串的总体描述。
详细的功能描述建议和上面的总体描述空一行分隔。
"""
from __future__ import barry_as_FLUFL   #注意，是
__all__ = []
import os
import sys
```

如果文件中包含 `__future__` 模块的导入，那么，该语句应该被放在所有导入的最前面。Python 的每个新版本都会增加一些新的功能，或者对原来的功能作一些改动。有些改动是不兼容旧版本的，也就是在当前版本运行正常的代码，到下一个版本运行就可能不正常了。从 Python 2.7 到 Python 3.x 就有不兼容的一些改动，比如 2.x 里的字符串用 'xxx' 表示 str，Unicode 字符串用 u'xxx' 表示 unicode，而在 3.x 中，所有字符串都被视为 unicode，因此，写 u'xxx' 和 'xxx' 是完全一致的，而在 2.x 中以 'xxx' 表示的 str 就必须写成 b'xxx'，以此表示"二进制字符串"。

要直接把代码升级到 3.x 是比较冒进的，因为有大量的改动需要测试。相反，可以在 2.7 版本中先在一部分代码中测试一些 3.x 的特性，如果没有问题，再移植到 3.x 不迟。Python 提供了 `__future__` 模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性。为了适应 Python 3.x 的新的字符串的表示方法，在 2.7 版本的代码中，可以通过 unicode_literals 来使用 Python 3.x 的新的语法：

```python
# still running on Python 2.7
from __future__ import unicode_literals
```

## frozenset() 用法

`frozenset()` 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。冻结集合为不可变对象，可以作字典的 key。

```python
>>> f = frozenset()
frozenset()   # 空的冻结集合

# 冻结集合不可使用 add / remove操作，但支持交、差、并集
>>> f |= {'foo'}
frozenset({'foo'})

# 冻结集合为不可变对象，可以作字典的key
>>> d = {f: 1}
{frozenset({'foo'}): 1}
```

`f |= {0,1}` 不会触发异常。frozenset 可以和普通 set 这样运算，对的。


## 空格

1. 一行只写一条语句，方便调试
3. 对等操作符：赋值，and，or，比较，计算等双目操作符前后加空格；
4. 逗号/分号后面加空格，前面不用加，print（a, b, c）；
5. `.` 前后不加空格，self.add（）；括号前后不加空格；
6. 行内注释 `#` 后面要有空格
7. `*`、`**` 等作为操作符时，前后可以加空格，但若和更低优先级的操作符同时使用并且不涉及括号，则建议前后不加空格。
8. 函数参数类型定义时，冒号前不应使用空格，冒号后需要加空格。函数返回值类型定义 `->` 前后添加空格。
9. 一行长度小于 120 个字符，换行用 `\`

```python
total = radix*2 - 1   # 符合，这里的操作符"*"前后不建议加空格

def create(self, name=None):   # 符合，参数默认值以及调用函数传递参数时使用的等号，前后不加空格
    self.create(name="mike")   # 无注解的默认值等号前后无空格

def create(self, input: name  =  None):   # 符合，有注解时，: 只是后面加空格，等号也加空格
def sample(constant: int) -> str:   # 符合，：后面加空格。->前后加空格
    pass

a = ((b + c) * d - 5) * 6   # 符合，多重括号内不加空格。
```

## 空行

1. 相对独立的程序块之间、变量说明之后必须加空行
2. 加载模块必须分开每个模块占一行，一行导入一个模块

## 大小写命名

- 注释必须与其描述的代码保持同样的缩进，并放在其上方相邻位置，此时不用缩进
- 类及接口，公共函数的文档字符串写在类与公共函数的下一行，有四个空格

2. 包、模块、函数、方法、函数参数，变量采用小写加下划线 "lower_with_under"；
3. 类+异常 采用大驼峰 "CapWords" -- 名字使用意义完整的英文描述，采用大写字母开头的单词（CapWords）风格命名
4. 常量采用大写加下划线 "CAPS_WITH_UNDER"。
5. 类或对象的私有成员一般用单下划线 `_` 开头；双下划线 "`__`" 开头的成员会被解释器自动改名，加上类名作为前缀，其作用是防止在类，继承场景中出现名字冲突，并不具有权限控制的作用，外部仍然可以访问

## 注意1

- 与 None 作比较要使用 "is" 或 "is not"，不要使用等号
- 传递实例类型参数后，函数内应使用 isinstance 函数进行参数检查，不要使用 type

- 使用推导式代替重复的逻辑操作构造序列。但推导式必须考虑可读性，不在一个推导式中使用三个以上的 for 语句
- 尽量不使用 for i in range(x) 的方式循环处理集合数据，而应使用 for x in iterable 的方式
- 如果一定要修改 sys.path，建议使用 append、extend 等方法加在 sys.path 最后，保证不影响其他标准库、三方包模块加载，避免在代码中修改 sys.path 列表
- assert 语句通常只在测试代码中使用，禁止在生产版本中包含 assert 功能
- 在 list 成员个数可以预知的情况下，创建 list 时需预留空间正好容纳所有成员的空间 -- 性能
- 在成员个数及内容皆不变的场景下尽量使用 tuple 替代 list
- PEP8 规范：对于 list，str，tuples，为空判断时使用 false，不为空使用 not 变量，不适用 ==[]
- print('6' * 3)，输出 666，是字符串；`*` 有特殊含义
- print((4+8)/2) 输出 2.0，为浮点数
- `//`、`/` 两个除法运算符：`/` 返回一个 float 数，一定会带小数点。`//` 不管操作数为何种数值类型，总是会舍去小数部分，返回数字序列中比真正的商小的最接近的数，负数的话，向更小方向。

```python
a=-7//3
b=7//3
print(a,type(a))   # -3,int
print(b,type(b))   # 2,int
a=-8//3
b=8//3
print(a,type(a))   # -3 int
print(b,type(b))   # 2 int
```

- 验证路径之前应该先将其标准化，使用 os.path.realpath("test") 的 realpath 函数，不是 abspath（）函数

## 字符串操作（格式化与拼接）

- 格式化类：%、format()、template
- 拼接类：+、()、join()
- 插值类：f-string，这种方式在可读性上秒杀 format() 方式，处理长字符串的拼接时，速度与 join() 方法相当。

- 因 python 中的字符串是不可变的类型，所以使用 " + " 号连接会生成一个新字符串，同时也重新申请了一段内存。在循环中，使用 format 方法、"%" 操作符和 join 方法代替 "+" 和 "+=" 操作符来完成字符串格式化【"+" 和 "+=" 会生成新的字符串】

```python
s1, s2 = 'Xi', 'Gua'
d='hi' + s1 + s2
# hiXiGua

# format()方法是python最推荐的字符串格式化的方法。
s1, s2 = 'Xi', 'Gua'
b='hi,{0}{1}'.format(s1,s2)
# hi, XiGua

# join方式拼接效率最高，使用略微复杂，但对于多个字符串进行拼接时，效率很高，只会有一次内存的申请。所以很擅长对列表的处理。
# 速度比较：非循环中速度对比：f-string > + > % > format

# %是字符串格式化方法，同样也能实现字符串拼接
s1, s2 = 'Xi', 'Gua'
c='hi, %s%s' %(s1, s2)
# hi, XiGua
```

```python
a=['HELLO']
b='HELLO'
c={"a":"123"}
print(a,b,c)
# ['HELLO'] HELLO {'a': '123'}
# 字符串的逗号不在打印，但list，dict还是要打印，所有的双引号变为单引号。
# a,b,c之间有一个空格隔开。
```

## 注意2（安全）

1. 因此根据红线要求，在 python 中不使用的功能、模块、函数、变量等一定要在代码中彻底删除，不给安全留下隐患。
2. 使用 os.path 库中的方法代替字符串拼接来完成文件系统路径的操作，屏蔽不同操作系统的差异
3. 使用 subprocess 模块代替 os.system 模块来执行 shell 命令
4. 建议使用 with 语句操作文件，简洁，规避 try 等异常
5. 防范会话固定的最佳实践：用户成功登陆后，应立即使原有会话 ID 失效，并建立一个新的会话 ID
6. 安全规范：
   - 缓冲区溢出，java 风险较小
   - 代码安全那几个检视方法也最好去了解清楚，也就是静态工具扫描、关键字搜索和自上而下分析法这三种
7. 禁止使用 subprocess 模块中的 shell= 选项，shell 参数置为 False，前面的参数转成 list 列表。注意：第 1 个参数中的 list 列表的第一个元素不允许是 "bash"、"cmd"、"/bin/sh"，第二个元素不允许是 "-c"；只有满足了这两个前提条件，参数 shell=False 的情况才是安全的
8. 禁止使用 eval 和 exec 执行不可信代码。Eval 将字符串当成 python 的表达式进行求值，并返回结果；exec 将字符串当成 python 代码进行执行。

```python
a=eval('1'+'2')
b=eval('1+2')
print(a,b)
# 12,3
```

9. 禁止调用 OS 命令解析器执行命令或运行程序，os.system 或 os.popen 经常被用来调用一个新的进程，使用标准的 API 替代操作系统的系统命令
10. Server 端发起网络请求前要验证是否存在：SSRF 漏洞（服务端请求伪 urlopen(url).info()，未进行参数检查。通过 SSRF 漏洞能实现扫描内网、修改应用报文实施 payload 攻击、拒绝服务攻击）做法：解析 URL，获取 host 看是否内网 IP，有跳转 URL，继续检查，有内网的 IP。
11. 不受信任的输入禁止使用 .format() 进行格式化，Python2.6 版本引入 "hi {0}".format("2012 lib") 的字符串格式化的方法。但该方法在特定场景下存在 python 沙箱逃逸，导致敏感信息泄露
12. 使用安全随机数，如果你需要一个真正的密码安全随机数，请使用 /dev/random 或者 urandom（win 环境）生成安全随机数；另外在 python 3.6 版本官方引入了一个 secrets 模块用于生成安全随机数

- 禁止抑制或者忽略已检查异常，例外情况：在资源释放失败不会影响程序后续行为的情况下，释放资源时发生的异常可以被抑制
- 禁止在异常中泄露敏感信息，例子：规定用户只能打开 /home/python/ 目录下的文件，用户不可能发现这个目录以外的任何信息
- 方法发生异常时要恢复到之前的对象状态：异常后面，或者 finally 后面要有 self.length -= PADDING
- 禁止使用 mktemp 创建临时文件，函数返回的临时文件名中含有进程 ID，可以用 NamedTemporaryFile()、mkstemp() 和 mkdtemp() 代替
- 临时文件使用完毕应及时删除，a = tempfile.NamedTemporaryFile(delete=True) 对的。并且在打开文件时用到了 delete 选项，使得文件在关闭时会被自动删除
- 在多用户系统中创建文件时指定合适的访问许可，with open("") 无法显示指定文件的访问权限。在多用户系统中存在问题。要使用 `os.fdopen(os.open('testfile.txt', flags, modes), 'w')`
- 控制解压文件占用的空间以避免压缩文件炸弹，禁止直接一次性递归解压压缩包全部内容，可以根据业务需要限制解压缩层次；解压前先判断文件大小


## 路径名

1. SQL 注入

```python
args = (id, type)
cur.execute('select id from xl_bugs where id = ? and type = ?', args)
# 还有，1）对于int做int转换，2）对于字符，用双引号替代单引号
```

2. 文件路径，要使用 os.realpath()，在所有平台对别名，软连接进行一致的处理；os.abspath() 可能会包含软链接，硬链接

```python
from pathlib import Path
```

```python
Path("demo.txt").replace("archive/demo.txt)
# def replace(self, target):  # 看replace的定义
# a. 把demo.txt内容覆盖到archive/demo.txt里面
# b. 把demo.txt拷贝到archive目录下,名字是demo.txt
# 结果: b
```

## 序列化

1. pickle.load、cPickle.load 和 shelve 模块加载不可信数据。根本原因：上面漏洞产生的根因是 reduce() 魔术方法，其作用是反序列化后产生的对象在结束时触发该函数，从而触发恶意行为。

规避方法：
- 用更高级的接口 `__getnewargs__()`、`__getstate__()`、`__setstate__()` 等代替 `__reduce__()` 魔术方法
- 进行反序列化操作之前，进行严格的过滤。

2. 禁止使用 yaml 模块的 load 函数，YAML 在数据序列化和配置文件中使用比较广泛，其在解析数据的时候遇到特定格式的数据类型会自动转换为 Python 的对象。建议使用 safe_load() 来加载

3. 禁止使用 jsonpickle 模块的 encode/decode 函数

4. 禁止使用 simplejson 模块的 scanstring 函数

5. 在序列化操作时建议使用较为安全的 json 模块，json.dumps 是安全的

```python
import json
d = dict(name='Bob', age=20, score=88)
json_str = json.dumps(d)
print(json_str)
```

对于 class 的示例对象，使用 json 序列化需要提供专门的转换函数

5. 模块解析 xml 文件时务必要使用参数 resolve_entities=False

- 生产代码不能包含任何调试入口点，由于调试或者测试目的，开发者经常在代码中添加特定的调测代码
- 禁止从第 3 方源下载并使用软件包
- 禁止在日志中保存口令、密钥等敏感信息
- 禁止使用私有或者弱加密算法
- 基于哈希算法的口令安全存储必须加入盐值（salt）
- 禁止将敏感信息硬编码在程序中
- 使用 SSLSocket 代替 Socket 来进行安全数据交互
- 对子类继承的变量要做显式定义和赋初值
- 禁止通过注释的方式删除无用的功能代码
- Python2.X 版本慎用 del 方法
- 代码发布前务必删除开发者信息及包含开发者信息的注释内容
- 保存外来不可信数据前要先转义

- 命令执行的字符串不要去拼接输入的参数，如果必须拼接时，要对输入参数进行白名单过滤
- 保证格式化字符串的正确性，例如：int 类型参数的拼接，对于参数要用 %d，不能用 %s
- 对传入的参数要做类型校验，例如：整数数据，可以对数据进行整数强制转换

## 通过代码覆盖分析进行测试补充

1. 语句覆盖：将每一条可执行语句都覆盖。百分百不等于看护所有修改。
2. 判定覆盖：每个判断语句的分支结果（为什么满足判定覆盖不等于满足条件覆盖？是因为有可能同一个条件的真假都导向同一个分支吗？）都得到至少一次覆盖。百分百不等于看护所有修改。判定覆盖一定满足语句覆盖。
3. 条件覆盖：每个条件的真假都能至少覆盖一次。条件覆盖不能保证判定覆盖。
4. 判定/条件覆盖：同时满足判定覆盖和条件覆盖。
5. 条件组合覆盖：每个判定的条件取值组合至少能被覆盖一次。eg 条件 1 与条件 2 的真假互相组合共四种。不是所有条件组合都能覆盖（eg if(c == 1 || c == 2) 不能同时满足）。满足条件覆盖、判定覆盖和语句覆盖，但是不能保证路径覆盖。
6. 路径覆盖：每一条路径都覆盖。不一定满足条件组合覆盖。

开发者测试的依据永远是需求而不是代码自身。

## 逗号、括号与字符串（易错题）

```python
>>> x = ('1', '22', '333')
>>> y = '1' '22' '333'   #这里怎么解释，实际y='122333'，是个str
>>> z = """1
22"""   #换行符也算一个字节长度
len(x)   # 3
len(y)   # 6
len(z)   # 4
```

**有逗号的情况**

```python
>>> y = '1', '22', '333'
('1', '22', '333')   # 默认输出一个元组，长度为3
>>> y1 = ['1', '22', '333']
['1', '22', '333']   # 输出一个列表
>>> y2 = {'1', '22', '333'}
{'1', '22', '333'}   # 输出一个集合
>>> y3 = ('1', '22', '333')
('1', '22', '333')   # 输出一个元组，注意括号没有起作用
```

Note：多个字符串用逗号分隔时，会默认转化为元组；不用逗号隔开时，则会自动合并为一个字符串

**无逗号的情况（另类写法）**

```python
>>> z = '1' '22' '333'
'122333'   # 默认输出一个字符串
>>> z1 = ['1' '22' '333']
['122333']   # 输出一个列表
>>> z2 = {'1' '22' '333'}
{'122333'}   # 输出一个集合
>>> z3 = ('1' '22' '333')
'122333'   # 输出一个字符串, 注意只有括号实现了字符串正常打印
```

```python
# 另类写法
>>> x = "1"
      "22"
      "333"
# IndentationError: unexpected indent
>>> x1 = ["1"
      "22"
      "333"]
['122333']   # 默认输出一个列表
>>> x2 = {"1"
      "22"
      "333"}
{'122333'}   # 输出一个集合
>>> x3 = ("1"
      "22"
      "333")
'122333'   # 输出一个字符串, 注意只有括号实现了字符串正常打印
```

## getrefcount

```python
import sys
x=[1,2]
print(sys.getrefcount(x))

def foo(a):
    print(sys.getrefcount(a))
    def fnc():
        b=a   # 增加b这一个引用。
        print(sys.getrefcount(b))
        print(sys.getrefcount(x))
    fnc()

foo(x)
print(sys.getrefcount(x))
# 2
# 4
# 5
# 5
# 2
```


## 运算符的优先级

Python 四种数字类型：int / long / float / complex

| 运算符 | 说明 | 优先级 | 结合性 |
| --- | --- | --- | --- |
| ** | 乘方 | 16 | 右 |
| ~ | 按位取反 | 15 | 右 |
| +（正号）、-（负号） | 符号运算符 | 14 | 右 |
| *、/、//、% | 乘除 | 13 | 左 |
| +、- | 加减 | 12 | 左 |
| >>、<< | 位移 | 11 | 左 |
| & | 按位与 | 10 | 右 |
| ^ | 按位异或 | 9 | 左 |
| \| | 按位或 | 8 | 左 |
| ==、!=、>、>=、<、<= | 比较运算符 | 7 | 左 |
| is、is not | is 运算符 | 6 | 左 |
| in、not in | in 运算符 | 5 | 左 |
| not | 逻辑非 | 4 | 右 |
| and | 逻辑与 | 3 | 左 |
| or | 逻辑或 | 2 | 左 |

- not > and > or
- is, is not > in, not in
- 乘方 ** > 按位取反 ~ > 正负号 > 乘除 > 加减 > 位移（>> <<）> 与 > 异或 > 或 > 比较 > is > in 等

Python 中 and 和 or 的用法：

- **and**：若一个表达式每一部分都为真，则返回最后一个真值，否则返回第一个假值
- **or**：若一个表达式每一部分都为假，则返回最后一个假值，否则返回第一个真值

```python
a=7//3.2  *2  >2
a=2.0 *2  >2
a=4.0>2
a=True
b=7//3.2  ** 2  >2
b=7//10.24  >2
b=0>2
b=False
a=3 or 5&8 ==1
a=3 or 0 ==1   # 5&8=0
a=3 or False
a=3   # （int类型）
```

## 元组

```python
a=1,2,3
b=(1)
c=(1,)
print(type(a),type(b),type(c))
# <class 'tuple'> <class 'int'> <class 'tuple'>
```

- 有无逗号是关键，一个元素在元组里面，必须要有逗号，否则不是元组；
- 有逗号就是元组。有无（）无所谓，（）python 编译器可能作为（）计算符，优先级很高。
- 注意，对于 {1}，{1,} 一样，都是 set 对象。

```python
a={"HELLO",}
b={"HELLO"}
c=("HELLO")
d=("HELLO",)   # d="HELLO",  #输出不变，有逗号，也是元组
print(type(a),type(b))   # <class 'set'> <class 'set'>
print(type(c),type(d))   # <class 'str'> <class 'tuple'>
```

2、元组解包：

```python
a,b=b,a   # 是个典型的元组解包,下面代码中若没有*d会报错，因为参数个数对不上，注意只有一个*参数，可以出现在前面，后面，中间。
a=1,2,3,4
b,c,*d=a
print(type(d),type(b),type(c))   # <class 'list'> <class 'int'> <class 'int'>
print(d,b,c)   # [3, 4] 1 2
```

## 列表

```python
l1 = ['Hello']
l22 = l1*3   # list里面的东西重复3, ['Hello', 'Hello', 'Hello']
l2 = [l1] * 3   # [['Hello'], ['Hello'], ['Hello']]
print(l22)
print(l2,l1)
l2[0][0] = 'World'   # l2[0]=['hello']，是个可变对象
print(l2)   # [['world'], ['world'], ['world']]  # 因为是*过来的，所以引用指向一个地方，如果修改其中一个，那么所有的都会修改
print(id(l2[0]),id(l2[1]))   # ID相等

print(l2[0],l2[1],l2[2])
# 全部指向一个对象，这个对象的value又指l2[0][0]='hello'
```

**list 删除方法：**

`list.pop(obj=list[-1])`：移除列表中的一个元素（默认最后一个元素），并且返回该元素的值

```python
l22 = ['Hello','123','1241']
print(l22)
c=l22.pop(1)   # 返回该元素的值，默认参数是最后一个,也可以填写下标
print(l22)
print(c,type(c))
# ['Hello', '123', '1241']
# ['Hello', '1241']
# 123 <class 'str'>
```

`list.remove(obj)`：移除列表中某个值的第一个匹配项。

```python
l22 = ['Hello','123','1241']
print(l22)
c=l22.remove('123')   # 该函数没有返回值，参数是值，值不存在则报错
print(l22)
print(c,type(c))
# ['Hello', '123', '1241']
# ['Hello', '1241']
# None <class 'NoneType'>
```

```python
l22 = ['Hello','123','1241']
print(l22)
del l22[1]   # 一定是下标，没有返回值
print(l22)
# ['Hello', '123', '1241']
# ['Hello', '1241']
```

**切片：**

```python
a="HELLO"
b=[123,345,234,124]
b[0:2]="sre"
a[0:2]="324"   # 报错，不可变对象
print(a,b)   # HELLO ['s', 'r', 'e', 234, 124]
```

```python
a="HELLO"
b=[123,345,234,124]
# b[0:2:2]="sre"   # 报错，当切片有步长时，二者个数要一样,且可以调整赋值
b[0:3:2]="sw"
print(a,b)   # HELLO ['s', 345, 'w', 124]
```

```python
b=[123,345,234,124]
# b[0:2:2]="sre"   # 报错，当切片有步长时，二者个数要一样,且可以调整赋值
b[1:1]="sw"   # 在1的位置插入s,w
print(b)   # [123, 's', 'w', 345, 234, 124]
```

**序列：**

- 支持 *，+ 操作的序列：字符串，list，元组。字典、集合不支持 *，+ 操作
- 元组支持切片读取，集合不支持切片读取

**5、字符串操作（split / find / index）**

```python
str.split(separator, maxsplit)
# 注意这里的第二个参数，当设置了maxsplit=2时，最多只会分隔两次，也就是生成3个组。
'a,b,c,d,e'.split(',', maxsplit=2)   # ['a', 'b', 'c,d,e']  分两次

# index在str中返回第一个匹配的到的位置，如果匹配不到返回ValueError
# find在str中返回第一个匹配的到的位置，匹配不到返回-1。
```

## 字典

```python
a={"a":12,"b":14}
for i in a:
    print(i)
# 1、此时没有指定是key或者value，缺省打印key，如果使用 a[key], 可以不存在会报 keyerror
# a
# b

sap_list = {'a': 'foo', 'b': 'bar'}
print('{a} {b}'.format(**sap_list))   # foo bar
# 此时解包为 v值
```

**2、get**

```python
a = {"a": 12, "b": 14}
c=a.get('a')
print(c)   # 12, 'a'不在则返回None，这个比a[]好，有对比，[]key不存在报错

c=a.get('d',"123")   # '123'为默认值
print(c)   # 123

c=a.setdefault("a",15)   # a存在，则返回12
print(c)
c=a.setdefault("c",15)   # c不存在，设置"c":15,返回15
print(c)
c=a.setdefault("c",16)   # 实际是读取，c已经存在，返回15
print(a)
# 可以直接读取任何键，不存在则自动新建一个key，避免抛出异常
```

**3、update**

```python
a = {"a": 12, "b": 14}
b = {"c":123}
a.update(b)   # b也是字典类型
print(a,b)
```

**删除**

```python
student = {'name':'zhangsan','age':20}
del student['name']   # key不存在，也会报错。
print(student)   # {'age': 20}

student = {'name':'zhangsan','age':20}
c=student.pop('name')   # 返回一个value，如果Key不存在，报错，也可以增加default值
print(student)   # {'age': 20}
print(c,type(c))   # zhangsan <class 'str'>

student = {'name':'zhangsan','age':20}
student.clear()
print(student)

student = {'name':'zhangsan','age':20}
c=student.popitem()   # 删除最后一个，而且把最后一个值作为元组返回。当字典为空时，调用popitem会报错
print(student)   # {'name': 'zhangsan'}
print(c,type(c))   # ('age', 20) <class 'tuple'>
# {'age': 20}
# {'age': 20}
# zhangsan <class 'str'>
# {}
# {'name': 'zhangsan'}
# ('age', 20) <class 'tuple'>
# 字典没有remove方法。
```

**pop 与 popitem 的区别：**

a、前者返回值，后者返回 (k,v) 元组
b、前者有参数（key），后者无参数

> 原文档练习：字典删除 key 的方法，错误的是 **A**
> - A. remove 【list方法，字典没有，输入'值'】
> - B. pop 【字典：返回一个值，输入参数是key，可不存在报错】【列表有：返回一个值，输入参数是小标，小标报错】
> - C. popitem 【返回一个元组，字典为空报错，列表没有popitem方法】
> - D. del 【都有，字典输入是key，list输入是小标】

**5、复制 copy**

```python
student = {'name':'zhangsan','age':20}
c=student.copy()
print(c,type(c),id(c))
print(c,type(c),id(student))
# {'name': 'zhangsan', 'age': 20} <class 'dict'> 69661440
# {'name': 'zhangsan', 'age': 20} <class 'dict'> 70496928
```

**小写 > 大写 > 数字的字典顺序是什么？**

```python
print('a'>'A' and 'A' > '1')   # True  # unittest从1，A，a顺序执行，从小到大
# a > A > 1
print('a'<'b')   # True
print('0'<'9')   # True
```


## 可变与不可变对象

**1、深浅拷贝的区别**

```python
>>> import copy
>>> a = [1, 2, ['x', 'y']]
>>> b = a             # 引用，全部影响
>>> c = copy.copy(a)  # 只是里面的可变对象有影响，copy浅拷贝：拷贝一个对象，但是对象的属性还是引用原来的。对于可变类型，比如列表、字典、集合，只是复制其引用。基于引用所作的改变会影响到被引用对象。
>>> d = copy.deepcopy(a)   # 完全拷贝一份，不影响
>>> a.append(3)
>>> a[2].append('z')
>>> a.append(['x', 'y'])
>>> print(a)   # [1, 2, ['x', 'y', 'z'], 3, ['x', 'y']]
>>> print(b)   # [1, 2, ['x', 'y', 'z'], 3, ['x', 'y']]
>>> print(c)   # [1, 2, ['x', 'y', 'z']]
>>> print(d)   # [1, 2, ['x', 'y']]
```

**2、L+L 与 +=L 的区别**

- `L = L + L` 这一步相当于对 L 重新赋值，L 的 id 改变，M 不再是 L 的引用。【对可变与不可变对象一样】
- `L += L`，只有当 L=[] 列表，或者 set 类型【可变对象】时，才有此特点 ID 不变，当 L 为 str，int 等类型时，ID 仍然会改变。

```python
L = [1,2]
M = L
L = L + L   # 这一步相当于对L重新赋值，L的id改变，M不再是L的引用。
L.append(1)
print(L, M)
# 【1,2,1，2,1】，【1,2】

L = [1,2]
M = L
L += L   # 注意，这里L+=L，L的ID没有改变
L.append(1)
print(L, M)
# [1, 2, 1, 2, 1] [1, 2, 1, 2, 1]
```

注意：`L = L + L` 与 `L += L` 的区别。特别注意只有当 L=[] 列表，或者 set 类型【可变对象】时，`L += L`，才有此特点 ID 不变，当 L 为 str，int 等类型时，ID 仍然会改变。

**3、可变与不可变对象在空值的区别**

```python
if () == ():
    print("() == ()")
if () is ():   # 不可变对象
    print("() is ()")
if [] == []:   # 可变对象
    print("[] == []")
if [] is []:
    print("[] is []")
# () == ()
# () is ()
# [] == []
```

可变对象的空值，ID 是不一样的，空可变对象 is 空可变对象 为 False；
不可变对象的空值，ID 是一样的。空不可变对象 is 空不可变对象 为 True；

copy 浅拷贝：拷贝一个对象，但是对象的属性还是引用原来的。对于可变类型，比如列表、字典、集合，只是复制其引用。基于引用所作的改变会影响到被引用对象。


## Decimal

```python
from decimal import Decimal,getcontext
print('%.5f' % 3.14)  # 3.14000
print(Decimal('3.145'))   # 3.145
getcontext().prec=8
print(Decimal(3.14))   # 3.140000000000000124344978758017532527446746826171875
print(Decimal(1)/Decimal(7))   # 0.14285714
print(Decimal(3))   # 3
```

```python
Decimal('1.135').quantize(Decimal('.01'))   # 1.1
Decimal('1.145').quantize(Decimal('.01'))   # 1.14
```

**四舍六入五成双**：a.bcd 保留小数点后两位，那么需要看小数点后第三位 d，如果 d > 5，那么直接进位；如果 d 小于 5，那么直接不进位。如果 d=5 的情况下，需要看 c，如果 c 是奇数，那么进位，如果 c 是偶数，那么不进位。

保留精度的 quantize 方法有一个默认参数 rounding。python 源码中可以看到：`quantize(self, exp, rounding=None, context=None)`，可追溯到 rounding 有一个默认值 `rounding=ROUND_HALF_EVEN`。

**各种四舍五入的方法**

- **ROUND_UP**：舍弃小数部分非 0 时，在前面增加数字，如 5.21 -> 5.3；-基础的

```python
print(Decimal(3.141).quantize(Decimal('.0001'),rounding=ROUND_UP))   # 3.1411
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_UP))     # 3.15
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_UP))    # -3.15
```

- **ROUND_DOWN**：舍弃小数部分，从不在前面数字做增加操作，如 5.21 -> 5.2；-基础

UP 和 DOWN 正负正常添加，不考虑正负。舍入方向为 0。

- **ROUND_CEILING**：如果 Decimal 为正，则做 ROUND_UP 操作；如果 Decimal 为负，则做 ROUND_DOWN 操作；----舍入方向到无穷大。

```python
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_CEILING)   # 3.15
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_CEILING)) # -3.14
```

- **ROUND_FLOOR**：如果 Decimal 为负，则做 ROUND_UP 操作；如果 Decimal 为正，则做 ROUND_DOWN 操作；----舍入方向到无穷小

```python
print(Decimal(-3.141).quantize(Decimal('.01'),rounding=ROUND_FLOOR))   # -3.15
print(Decimal(3.141).quantize(Decimal('.01'),rounding=ROUND_FLOOR))    # 3.14
```

CEILING 和 FLOOR，复数刚好反过来【注意正负】，整数 CEILING=UP，DOWN=FLOOR。

- **ROUND_HALF_DOWN**【标准的五舍六入，中间向下】：如果舍弃部分 > .5，则做 ROUND_UP 操作；否则，做 ROUND_DOWN 操作；

```python
print(Decimal('3.135').quantize(Decimal('.01'),rounding=ROUND_HALF_DOWN))  # 3.13
Decimal('-3.136').quantize(Decimal('.01'),rounding=ROUND_HALF_DOWN)         # -3.14
```

- **ROUND_HALF_UP**【标准的四舍五入，中间向上】：如果舍弃部分 >= .5，则做 ROUND_UP 操作；否则，做 ROUND_DOWN 操作；不考虑正负

- **ROUND_HALF_EVEN**：四舍五入，5 特别，如果最后一位是 5，则检查前一位，奇数在向上，偶数则向下

```python
print(Decimal('3.135').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.14; 到时第二位是奇数，则上一位，变为3.14
print(Decimal('3.145').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.14; 到时第二位是偶数，向下，直接保留3.14
print(Decimal('3.146').quantize(Decimal('.01'),rounding=ROUND_HALF_EVEN))  # 3.15; 直接五入，变为3.15
```

其次还有"四舍六入五成双"总结（银行家舍入法）。

## CProfile

- **ncalls**: 函数调用次数
- **tottime**: 函数执行时间，不包含子函数调用时间，如 this_is_a_time_consume_task 的统计不包含 time.sleep 的时间
- **percall**: tottime/ncalls
- **cumtime**: 包含子函数调用时间

另外：可以通过在上述命令中通过 `-s` 指定排序，如下面的命令按照 cumtime 排序。

```python
p = Profile()
p.runcall(main)
p.print_stats()
```

需要使用侵入式的手段。我们仍然使用上面脚本为例，我们只想统计 main 的执行情况。

cProfile 比 profile 快。

**关于 pylint 正确的说法是：**

1. too many instance attributes 规定类属性不得多于 7 个
2. too many arguments 告警可以通过 dict class 等方式消除
3. deprecated-lambda 告警的消除做法是用列表推导代替 map 和 filter
4. 遍历设计范围和索引时尽量使用 enumerate 和 range

## Datatime

**1、datetime.date**

与 time 库一样，datetime 库也有获取当前日期的类，日历日期值用 datetime.date 表示。

- `datetime.date.today()`。

```python
from datetime import *
s=date.today()   # 2022-01-08
```

而 timetuple() 函数返回的是 time 库中常用的 time.struct_time 结构体，这样你就可以像使用 struct_time 结构体一样，获取单一的时间数据，不过因为 datetime.date.today() 只有日期，所以时间数据为 0

```python
from datetime import *
s=date.today()   # 2022-01-08
now = s.timetuple()
print(now)
# time.struct_time(tm_year=2022, tm_mon=1, tm_mday=8, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=5, tm_yday=8, tm_isdst=-1)
# tm_wday=5, #返回星期几（这个是0-6）,实际六，但返回5
print(today.isoweekday())
# tm_yday=8，今年过了xx天
```

返回日期的多边格里高利度序数，其中 1 年 1 月 1 日具有序数 1。如果 1 年 1 月 1 日具有序数 1，则 1 年 1 月 2 日将具有序数 2，依此类推。

Fromordinal 函数返回多边格里高利度序数对应的日期 datetime.date 对象

```python
print(today.toordinal())   # 738163
print(today.fromordinal(1))   # 0001-01-01，注意补齐0
```

获取当前日期完整数据，与 time.ctime() 类似，只是时间是 00:00:00

```python
print(today.ctime())   # Sat Jan  8 00:00:00 2022
```

获取星期几，两种方式：

```python
print(today.weekday())   # 5,【0-6】
print(today.isoweekday())   # 6 【1-7】
```

replace 将数字日期转换为 datetime.date 对象时间

```python
print(today.replace(2020, 10, 10))   # 2020-10-10
print(today.isoformat())   # 2022-01-08
# 返回年，该年的第几周以及周几,返回值元组类型
print(today.isocalendar())   # (2022, 1, 6)
# 将datetime.date对象时间转换为指定的字符串格式
print(today.strftime("%Y:%m:%d"))   # 2022:01:08,注意 补0
```

**2、datetime.time**

```python
t = time(19, 7, 20)
print(t)   # 19:07:20  注意补齐0
# 获取时间的最大值与最小值
print(t.min, t.max)   # 00:00:00 23:59:59.999999
# 获取当前输入时间的时，分，秒数据
print(t.hour, t.minute, t.second, t.microsecond, t.tzinfo)
# 时间分辨率，datetime.time被限制为整微秒值
print(t.resolution)   # 0:00:00.000001
# 替换时间值，返回datetime.time时间
print(t.replace(15, 30, 30))   # 15:30:30
# 输出指定格式时间的字符串
print(t.strftime("%H-%M-%S"))   # 19-20-20
```

**3、datetime.timedelta**

```python
today = date.today()
print("今天日期：", today)
one_day = timedelta(seconds=24*3600)
# one_day = timedelta(days=1)   # 与上面是一样，其实是个时间间隔
print(type(one_day))
yesterday = today - one_day
tomorrow = today + one_day
print("昨天日期", yesterday)
print("明天日期", tomorrow)
print("昨天与明天相差{0}天", (yesterday - tomorrow).days)
print("明天与昨天相差{0}天", tomorrow - yesterday)
# 今天日期： 2022-01-08
# <class 'datetime.timedelta'>
# 昨天日期 2022-01-07
# 明天日期 2022-01-09
# 昨天与明天相差{0}天 -2
# 明天与昨天相差{0}天 2 days, 0:00:00
```


## Random

1. `random.random()` 用于生成一个 0 到 1 的随机浮点数: 0 <= n < 1.0。不包括 1，前闭后开
2. `random.uniform`，`random.uniform(a, b)`，用于生成一个指定范围内的随机浮点数，两个参数其中一个是上限，一个是下限。如果 a > b，则生成的随机数 n: a <= n <= b。如果 a < b，则 b <= n <= a。其中 a 和 b 的顺序无关。前闭后闭
3. `random.randint(a, b)`，用于生成一个指定范围内的整数。其中参数 a 是下限，参数 b 是上限，生成的随机数 n: a <= n <= b，前闭后闭，注意这里如果 a，b 顺序颠倒，会报错：empty range for randrange() (2,2, 0)。`randint(a,b) == randrange(a, b+1)`
4. `random.randrange(0, 101, 2)`，用于生产指定范围内的整数，类似 range，可以有步长。注意是前闭后开，类似 range(a,b,step)
5. `random.sample(sequence, k)`，从指定序列中随机获取指定长度的片断。sample 函数不会修改原有序列

```python
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(list, 5)   # 从list中随机获取5个元素，作为一个返回
print(slice)   # [1, 2, 9, 3, 6]
print(list)   # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

6. `random.shuffle(x[, random])`，用于将一个列表中的元素打乱。如:

```python
p = ["Python", "is", "powerful", "simple", "and so on..."]
random.shuffle(p)
print(p)
# ['powerful', 'simple', 'is', 'Python', 'and so on...']

print(random.choices(['go', 'go', "stop"]，key=2))
# ['go'，'go']  ，返回列表
print(random.choice(['go', 'go', "stop"]))
# stop ，返回一个值
```

## Tracemalloc

```python
import tracemalloc
tracemalloc.start()
```

```python
s1 = tracemalloc.take_snapshot()
pass
s2 = tracemalloc.take_snapshot()
diff = s2.compare_to(s1, "lineno")
log.info("============ start print stat diff =============")
for stat in diff[:10]:
    # log the top 10 difference
    log.info(stat)
#
# d:\pythonPorject\1.py:10: size=612 B (+612 B), count=2 (+2), average=306 B
# d:\pythonPorject\1.py:8: size=68 B (+68 B), count=2 (+2), average=34 B
```

可以打印行号。具体解释：
- Count：内存块数 (int)。
- Size：内存块的总大小（以字节为单位 int）。

## 正则表达式

```python
import re
m = re.match('(\w\w\w)-(\d?)', 'abc-123')   # ?表示匹配前面0或者1个d数字
print(m.group())   # abc-1
print(m.groups())   # ('abc', '1')   #注意注意
print(m.group(0))   # abc-1
print(m.group(1))   # abc
print(m.group(2))   # 1
```

`re.group(0)` 与 `re.group()` 相同返回字符串。

```python
import re
line = "Cats are smarter than dogs"
# .* 表示任意匹配除换行符（\n、\r）之外的任何单个或多个字符
matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)
if matchObj:
    print("matchObj.groups() : ", matchObj.groups())
    print("matchObj.group() : ", matchObj.group())
    print("matchObj.group(0) : ", matchObj.group(0))
    print("matchObj.group(1) : ", matchObj.group(1))
    print("matchObj.group(2) : ", matchObj.group(2))
# matchObj.groups() :  ('Cats', 'smarter')
# matchObj.group() :  Cats are smarter than dogs
# matchObj.group(0) :  Cats are smarter than dogs
# matchObj.group(1) :  Cats
# matchObj.group(2) :  smarte
```

- `re.match` 只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回 None，而 `re.search` 匹配整个字符串，直到找到一个；
- `result = re.findall('.*?(\d+).*', content)`，返回一个 list，全部找到

**基本匹配：**

| 字符 | 描述 |
| --- | --- |
| $ | 匹配输入字符串的结尾位置。如果设置了 RegExp 对象的 Multiline 属性，则 $ 也匹配 '\n' 或 '\r'。要匹配 $ 字符本身，请使用 \$。 |
| ( ) | 标记一个子表达式的开始和结束位置。子表达式可以获取供以后使用【后面使用主要看()】。要匹配这些字符，请使用 \( 和 \)。 |
| * | 匹配前面的子表达式零次或多次。要匹配 * 字符，请使用 \*。 |
| + | 匹配前面的子表达式一次或多次。要匹配 + 字符，请使用 \+。 |
| . | 匹配除换行符 \n 之外的任何单字符。要匹配 . ，请使用 \. 。 |
| [ | 标记一个中括号表达式的开始。要匹配 [，请使用 \[。 |
| ? | 匹配前面的子表达式零次或一次，或指明一个非贪婪限定符。要匹配 ? 字符，请使用 \?。 |
| \ | 将下一个字符标记为或特殊字符、或原义字符、或向后引用、或八进制转义符。例如， 'n' 匹配字符 'n'。'\n' 匹配换行符。序列 '\\' 匹配 "\"，而 '\(' 则匹配 "("。 |
| ^ | 匹配输入字符串的开始位置，除非在方括号表达式中使用，当该符号在方括号表达式中使用时，表示不接受该方括号表达式中的字符集合。要匹配 ^ 字符本身使用 \^。 |
| { | 标记限定符表达式的开始。要匹配 {，请使用 \{。 |
| \| | 指明两项之间的一个选择。要匹配 \|，请使用 \|。 |

```python
import re
print(re.match('www', 'www.runoob.com').span())   # 在起始位置匹配 (0, 3)
print(re.match('com', 'www.runoob.com'))   # 不在起始位置匹配 None
```


## Unittest

```python
import unittest

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        print("up")
    def tearDown(self) -> None:
        print("down")
    def test_A(self):
        print("A")
    def test_a(self):
        print("a")
    def test_4(self):
        print("4")
    def test_3(self):
        print("3")
    def test_5(self):
        print("5")

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase("test_A"))   #执行顺序改变
    suite.addTest(MyTestCase("test_3"))   #执行顺序改变
    runner = unittest.TextTestRunner()    #此时setup，setdown让然生效
    runner.run(suite)
print(id(a),id(b))
# up
# A
# down
# .up
# 3
# down
# ----------------------------------------------------------------------
# Ran 2 tests in 0.005s
# OK
# <class 'unittest.suite.TestSuite'>   #可以打印，如果把runner，改为：unittest.main()，则打印不出来，奇怪。
```

- 测试用例的命名规则必须为 test_xxx，这样的才是用例才会自动执行
- 调用 `unittest.main()` 运行所有内容，然后程序结束了

**mock 的使用：**

```python
class Count():
    def add(self, a, b):
        return a + b

class MockDemo(unittest.TestCase):
    def test_add(self):
        count = Count()
        count.add = mock.Mock(return_value=13, side_effect=count.add) #
        result = count.add(8, 8)
        print(result)
        count.add.assert_called_with(8, 8)
        self.assertEqual(result, 16)

if __name__ == '__main__':
    unittest.main()
```

注意：`count.add = mock.Mock(return_value=13, side_effect=count.add)`，有了 add 函数，mock 的返回值不生效了。

## numpy

```python
int_list * 2
# 输出结果：
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

这就是 NumPy 的用武之地。NumPy 是专为简化 Python 中的数组运算而设计的。我们可以快速将整数列表转换为一个 NumPy 数组：

```python
import numpy as np
print(np.__version__)
int_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
int_arr = np.array(int_list)
print(int_arr)
print(int_arr * 2)
# 1.19.5
# [0 1 2 3 4 5 6 7 8 9]
# [ 0  2  4  6  8 10 12 14 16 18]
```

而且，每个 NumPy 数组都具有以下属性：在 numpy 中，ndim 表示维数，shape 表示每一维的大小，size 表示数组中元素的总数，dtype 表示数组的数据类型（例如 int、float、string 等）。

```python
# int_arr ndim:  1
# int_arr shape:  (10,)
# int_arr size:  10
# int_arr dtype:  int32
```

py 的 arr 也支持切片操作。

创建多维数组：

```python
import numpy as np
arr_2d = np.zeros((3, 5))
print(arr_2d)
# [[0. 0. 0. 0. 0.] 
#  [0. 0. 0. 0. 0.] 
#  [0. 0. 0. 0. 0.]]
```

## PDB

1. q(uit) 退出调试器。被执行的程序将被中止。
2. s(tep) 运行当前行，在第一个可以停止的位置（在被调用的函数内部或在当前函数的下一行）停下。
3. n(ext) 继续运行，直到运行到当前函数的下一行，或当前函数返回为止。（next 和 step 之间的区别在于，step 进入被调用函数内部并停止，而 next（几乎）全速运行被调用函数，仅在当前函数的下一行停止。）
4. unt(il) [lineno] 如果不带参数，则继续运行，直。
5. r(eturn) 继续运行，直到当前函数返回。
6. c(ont(inue)) 继续运行，。
7. j(ump) lineno 设置即将运行的下一行。仅可用于堆栈最底部的帧。它可以往回跳来再次运行代码，也可以往前跳来跳过不想运行的代码。
8. l(ist) [first[, last]] 列出当前文件的源代码。如果不带参数，则列出当前行周围的 11 行，或继续前一个列表。如果用 . 作为参数，则列出当前行周围的 11 行。如果带有一个参数，则列出那一行周围的 11 行。如果带有两个参数，则列出所给的范围中的代码；如果第二个参数小于第一个参数，则将其解释为列出行数的计数。

**断点：**

a. 一行只能一个断点 – 可以有多个断点 yes

b. 断点可以配置到 import 位置 -- 是可以的 yes

c. 断点只能配置已经加载的文件 ok

d. 断点可以配置到空行 – 错误

## Log

默认情况下，logging 将日志打印到屏幕，日志级别为 WARNING；

日志级别大小关系为：`CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET`，也可以自己定义日志级别。当设置的 level 高于打印的时候指定的 level，则不打印。

华为 log 规范：

## Type() 创建类

```python
class ListMetaClass(type):
    def __new__(cls, name, bases, attrs):
        print(name)
        print(bases)
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)

class DefineList(list, metaclass=ListMetaClass):
    pass

defineList = DefineList()
defineList.add(1)
```

- `DefineList`，类名
- `(<class 'list'>,)`，基类

**元类：** 有 `__metaclass__` 这个属性吗？如果是，Python 会在内存中通过 `__metaclass__` 创建一个名字为 Foo 的类对象（我说的是类对象，请紧跟我的思路）。如果 Python 没有找到 `__metaclass__`，它会继续在 Bar（父类）中寻找 `__metaclass__` 属性，并尝试做和前面同样的操作。如果 Python 在任何父类中都找不到 `__metaclass__`，它就会在模块层次中去寻找 `__metaclass__`，并尝试做同样的操作。如果还是找不到 `__metaclass__`，Python 就会用内置的 type 来创建这个类对象。

现在的问题就是，你可以在 `__metaclass__` 中放置些什么代码呢？答案就是：可以创建一个类的东西。那么什么可以用来创建一个类呢？type，或者任何使用到 type 或者子类化 type 的东东都可以。

```python
yourclass = type("YourClass",(),{})
class A():
    pass
class B(A):
    pass
print(type(B))
# type
# 所有类的type都是type
```

## 几类工具

- A. flake8 # 代码规范检查
- B. yapf # 格式化工具
- C. AutoPep8 # 格式化工具
- D. coverage # 代码覆盖率测试

## 性能优化

频繁调用的外部对象，用局部变量引用

```python
import math
def afun(tan=math.tan):
    # 避免重复找全局变量/对象tan，频繁调用的外部对象，用局部变量引用
    for x in xrange(10000):
        return tan(x)
```

## Python 链式比较

```python
4>3==3   # True      # 4>3 and 3==3
```

记住答案：JustAJoke123。


## 疑问整理

**疑问1：**

```python
class A:
    a = 10
    b = ['12', 2, 12]
    print(a)                        # 10
    print(b)                        # ['12', 2, 12]
    print([a * element for element in b])   # NameError: name 'a' is not defined
A()
# NameError: name 'a' is not defined
```

**疑问2：**

```python
a = (1, 2, 3)   # 原则，为不可变对象，a，b的ID是一样的；可变对象，则ID是不一样的。
b = a[::]
print(id(a), id(b))
```

**疑问3：**

```python
print('a,b,c,d,e'.split(',', maxsplit=2))
# ['a', 'b', 'c,d,e']  分两次
```

**疑问4：** 以下内容介绍了几种比较复杂的触发异常情景：

- 如果执行 try 子句期间触发了某个异常，则某个 except 子句应处理该异常。如果该异常没有 except 子句处理，在 finally 子句执行后会被重新触发。
- except 或 else 子句执行期间也会触发异常。同样，该异常会在 finally 子句执行之后被重新触发。
- 如果 finally 子句中包含 break、continue 或 return 等语句，异常将不会被重新引发。
- 如果执行 try 语句时遇到 break、continue 或 return 语句，则 finally 子句在执行 break、continue 或 return 语句之前执行。
- 如果 finally 子句中包含 return 语句，则返回值来自 finally 子句的某个 return 语句的返回值，而不是来自 try 子句的 return 语句的返回值。

**疑问5：**

iter 对象的第一个参数如果是可调用对象时，会一直调用该对象直到与第二个参数相同或者 raise StopIteration

```python
class Next(object):
    def __init__(self):
        self.data = [0, 1, 20, 3, 40]
        self._inter = iter(self.data)
    def getLen(self):
        return len(self.data)
    def __iter__(self):
        return self
    def __call__(self):
        return next(self._inter)
    def __next__(self):
        return next(self._inter)

for it in iter(Next(), 20):   # 20不会出现，切记。。。
    print(it)
# 0
# 1
```
