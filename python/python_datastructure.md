python算法复杂度列表：

https://wiki.python.org/moin/TimeComplexity

### 乱序字符串检查

乱序字符串是指一个字符串只是另一个字符串的重新排列。例如，'heart' 和 'earth' 就是乱序字符串。'python' 和 'typhon' 也是。为了简单起见，我们假设所讨论的两个字符串具有相等的长度，并且他们由 26 个小写字母集合组成。我们的目标是写一个布尔函数，它将两个字符串做参数并返回它们是不是乱序。

解法1：检查

```python
def anagramSolution1(s1,s2):
	alist = list(s2)
    
	pos1 = 0
	stillOK = True
    
	while pos1 < len(s1) and stillOK:
		pos2 = 0
		found = False
		while pos2 < len(alist) and not found:
			if s1[pos1] == alist[pos2]:
				found = True
			else:
				pos2 = pos2 + 1
                
		if found:
			alist[pos2] = None
		else:
			stillOK = False
            
		pos1 = pos1 + 1
        
	return stillOK

print(anagramSolution1('abcd','dcba'))
```

算法复杂度：

$\sum_{i=1}^n i = n(n+1)/2=\frac12n^2+\frac12n$

$O(n^2)$

解法2：排序和比较

```python
def anagramSolution2(s1,s2):
	alist1 = list(s1)
	alist2 = list(s2)
	alist1.sort()
	alist2.sort()
	pos = 0
	matches = True
	while pos < len(s1) and matches:
		if alist1[pos]==alist2[pos]:
			pos = pos + 1
		else:
			matches = False
	return matches
print(anagramSolution2('abcde','edcba'))
```

算法复杂度O(n)+排序复杂度O(n^2)或O(nlogn)

解法3：穷举法

对于乱序检测，我们可以生成 s1 的所有乱序字符串列表，然后查看是不是有 s2。这种方法有一点困难。当 s1 生成所有可能的字符串时，第一个位置有 n 种可能，第二个位置有 n-1 种，第三个位置有 n-3 种，等等。总数为 n∗(n−1)∗(n−2)∗...∗3∗2∗1n∗(n−1)∗(n−2)∗...∗3∗2∗1， 即 n!。虽然一些字符串可能是重复的，程序也不可能提前知道这样，所以他仍然会生成 n! 个字符串。
事实证明，n! 比 n^2 增长还快，事实上，如果 s1 有 20个字符长，则将有 20! =2,432,902,008,176,640,000 个字符串产生。如果我们每秒处理一种可能字符串，那么需要77,146,816,596 年才能过完整个列表。所以这不是很好的解决方案。

解法4: 计数和比较

```python
def anagramSolution4(s1,s2):
    c1 = [0]*26
    c2 = [0]*26

    for i in range(len(s1)):
        pos = ord(s1[i])-ord('a')
        c1[pos] = c1[pos] + 1

    for i in range(len(s2)):
        pos = ord(s2[i])-ord('a')
        c2[pos] = c2[pos] + 1

    j = 0
    stillOK = True
    while j<26 and stillOK:
        if c1[j]==c2[j]:
            j = j + 1
        else:
            stillOK = False

    return stillOK

print(anagramSolution4('apple','pleap'))
```

T(n)=2n+26T(n)=2n+26，即O(n)，线性量级

虽然最后一个方案在线性时间执行，但它需要额外的存储来保存两个字符计数列表。换句话说，该算法牺牲了空间以获得时间。

### 列表

两个常见的操作是索引和分配到索引位置。无论列表有多大，这两个操作都需要相同的时间。当这样的操作和列表的大小无关时，它们是 O（1）。

另一个非常常见的编程任务是增加一个列表。有两种方法可以创建更长的列表，可以使用append 方法或拼接运算符。append 方法是 O（1)。 然而，拼接运算符是 O（k），其中 k是要拼接的列表的大小。

```python
def test1():
    l = []
    for i in range(1000):
        l = l + [i]

def test2():
    l = []
    for i in range(1000):
        l.append(i)

def test3():
    l = [i for i in range(1000)]

def test4():
    l = list(range(1000))
```

```python
t1 = Timer("test1()", "from __main__ import test1")
print("concat ",t1.timeit(number=1000), "milliseconds")
t2 = Timer("test2()", "from __main__ import test2")
print("append ",t2.timeit(number=1000), "milliseconds")
t3 = Timer("test3()", "from __main__ import test3")
print("comprehension ",t3.timeit(number=1000), "milliseconds")
t4 = Timer("test4()", "from __main__ import test4")
print("list range ",t4.timeit(number=1000), "milliseconds")

concat  6.54352807999 milliseconds
append  0.306292057037 milliseconds
comprehension  0.147661924362 milliseconds
list range  0.0655000209808 milliseconds
```

当列表末尾调用 pop 时，它需要 O(1), 但是当在列表中第一个元素或者中间任何地方调用 pop,它是 O(n)。原因在于 Python 实现列表的方式，当一个项从列表前面取出，列表中的其他元素靠近起始位置移动一个位置。你会看到索引操作为 O(1)。

| Operation        | Big-O Efficiency |
| ---------------- | ---------------- |
| index []         | O(1)             |
| index assignment | O(1)             |
| append           | O(1)             |
| pop()            | O(1)             |
| pop(i)           | O(n)             |
| insert(i,item)   | O(n)             |
| del operator     | O(n)             |
| iteration        | O(n)             |
| contains (in)    | O(n)             |
| get slice [x:y]  | O(k)             |
| del slice        | O(n)             |
| set slice        | O(n+k)           |
| reverse          | O(n)             |
| concatenate      | O(k)             |
| sort             | O(n log n)       |
| multiply         | O(nk)            |

### 字典

| operation     | Big-O Efficiency |
| ------------- | ---------------- |
| copy          | O(n)             |
| get item      | O(1)             |
| set item      | O(1)             |
| delete item   | O(1)             |
| contains (in) | O(1)             |
| iteration     | O(n)             |

列表的 contains 操作符是 O(n)，字典的 contains 操作符是 O(1)。

