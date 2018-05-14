## Pandas库
Numpy  
基础数据类型，关注数据的结构表达，维度：数据间关系  
Pandas  
扩展数据类型，关注数据的应用表达，维度：数据与索引间关系

### Series类型  
Series类型由一组数据及与之相关的数据索引组成  
Series类型可以由如下类型创建：  
• Python列表,index与列表元素个数一致  
b = pd.Series([9,8,7,6],index=['a','b','c','d'])    

• 标量值,index表达Series类型的尺寸  
s = pd.Series(25, index=['a','b','c'])  

• Python字典，键值对中的"键"是索引，index从字典中进行选择操作    
e = pd.Series({'a':9,'b':8,'c':7},index=['c','a','b','d'])  

• ndarray,索引和数据都可以通过ndarray类型创建    
m = pd.Series(np.arange(5), index=np.arange(9,4,-1))  

• 其他函数，range()函数等  

Series类型包括index和values两部分 
   b[['c','d','a']]

Series类型的操作类似ndarray类型 
• 索引方法相同，采用[]  
• NumPy中运算和操作可用于Series类型  
• 可以通过自定义索引的列表进行切片  
• 可以通过自动索引进行切片，如果存在自定义索引，则一同被切片  

Series类型的操作类似Python字典类型  
• 通过自定义索引访问  
• 保留字in操作  
• 使用.get()方法  
b.get('f',100)

Series+Series  
Series类型在运算中会自动对齐不同索引的数据  

Series对象和索引都可以有一个名字，存储在属性.name中  

### DataFrame类型  
DataFrame类型由共用相同索引的一组列组成  

DataFrame类型可以由如下类型创建：  
• 二维ndarray对象   
d = pd.DataFrame(np.arange(10).reshape(2,5))

• 由一维ndarray、列表、字典、元组或Series构成的字典  
```
dt = {'one':pd.Series([1,2,3],index=['a','b','c']),
      'two':pd.Series([9,8,7,6],index=['a','b','c','d'])}  
d = pd.DataFrame(dt)  
pd.DataFrame(dt,index=['b','c','d'],columns=['two','three'])

d1 = {'one':[1,2,3,4],'two':[9,8,7,6]}
d = pd.DataFrame(d1,index=['a','b','c','d'])
```
• Series类型  

• 其他的DataFrame类型  

### Pandas库的数据类型操作  
如何改变Series和DataFrame对象？  
增加或重排：重新索引  
.reindex()能够改变或重排Series和DataFrame索引   
.reindex(index=None, columns=None, …)的参数  
index, columns 新的行列自定义索引  
fill_value 重新索引中，用于填充缺失位置的值  
method 填充方法, ffill当前值向前填充，bfill向后填充  
limit 最大填充量  
copy 默认True，生成新的对象，False时，新旧相等不复制  
newc = d.columns.insert(4,'新增')  
newd = d.reindex(columns=newc,fill_value=200)

Series和DataFrame的索引是Index类型  
Index对象是不可修改类型  

索引类型的常用方法  
.append(idx) 连接另一个Index对象，产生新的Index对象  
.diff(idx) 计算差集，产生新的Index对象  
.intersection(idx) 计算交集  
.union(idx) 计算并集  
.delete(loc) 删除loc位置处的元素  
.insert(loc,e) 在loc位置增加一个元素e  

删除：drop  
.drop()能够删除Series和DataFrame指定行或列索引  

### Pandas库的数据类型运算  
算术运算法则  
算术运算根据行列索引，补齐后运算，运算默认产生浮点数  
补齐时缺项填充NaN (空值)  
二维和一维、一维和零维间为广播运算,一维Series默认在轴1参与运算，用axis=0可以领一维Series参与轴0运算    
采用+ ‐ * /符号进行的二元运算产生新的对象  

方法形式的运算  
.add(d, **argws) 类型间加法运算，可选参数  
.sub(d, **argws) 类型间减法运算，可选参数  
.mul(d, **argws) 类型间乘法运算，可选参数  
.div(d, **argws) 类型间除法运算，可选参数  

a = pd.DataFrame(np.arange(12).reshape(3,4))  
b = pd.DataFrame(np.arange(20).reshape(4,5))  
b.add(a,fill_value = 100)  
a.mul(b,fill_value = 0)  

比较运算法则  
比较运算只能比较相同索引的元素，不进行补齐  
二维和一维、一维和零维间为广播运算，默认在1轴  
采用> < >= <= == !=等符号进行的二元运算产生布尔对象  

### 数据的特征分析
>Pandas库的数据排序  

.sort_index()方法在指定轴上根据索引进行排序，默认升序  
.sort_index(axis=0, ascending=True)  

.sort_values()方法在指定轴上根据数值进行排序，默认升序  
Series.sort_values(axis=0, ascending=True)  
DataFrame.sort_values(by, axis=0, ascending=True)  
by : axis轴上的某个索引或索引列表  
NaN统一放到排序末尾

>数据的基本统计分析  

基本的统计分析函数  
适用于Series和DataFrame类型  
.sum() 计算数据的总和，按0轴计算，下同  
.count() 非NaN值的数量  
.mean() .median() 计算数据的算术平均值、算术中位数  
.var() .std() 计算数据的方差、标准差  
.min() .max() 计算数据的最小值、最大值  

适用于Series类型  
.argmin() .argmax() 计算数据最大值、最小值所在位置的索引位置（自动索引）  
.idxmin() .idxmax() 计算数据最大值、最小值所在位置的索引（自定义索引）  

适用于Series和DataFrame类型  
.describe() 针对0轴（各列）的统计汇总  

>数据的累计统计分析  

累计统计分析函数  
适用于Series和DataFrame类型，累计计算  
.cumsum() 依次给出前1、2、…、n个数的和  
.cumprod() 依次给出前1、2、…、n个数的积  
.cummax() 依次给出前1、2、…、n个数的最大值  
.cummin() 依次给出前1、2、…、n个数的最小值  

适用于Series和DataFrame类型，滚动计算（窗口计算）  
.rolling(w).sum() 依次计算相邻w个元素的和  
.rolling(w).mean() 依次计算相邻w个元素的算术平均值  
.rolling(w).var() 依次计算相邻w个元素的方差  
.rolling(w).std() 依次计算相邻w个元素的标准差  
.rolling(w).min() .max() 依次计算相邻w个元素的最小值和最大值  

>数据的相关分析  

适用于Series和DataFrame类型  
.cov() 计算协方差矩阵  
.corr() 计算相关系数矩阵, Pearson、Spearman、Kendall等系数  
