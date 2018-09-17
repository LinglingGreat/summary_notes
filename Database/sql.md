语法顺序：

Select [distinct]

from

where 

group by

having

union

order by

语句执行顺序：

from

where

group by

having

Select

distinct

union

order by

where条件中不能跟聚合函数，而having后面可以；执行顺序where>聚合函数(sum,min,max,avg,count)>having 

https://blog.csdn.net/moguxiansheng1106/article/details/44258499

## 基础

```sql
SQL SELECT Column_ FROM Mytable
SQL DISTINCT(放在SELECT后使用)
SQL WHERE(设置条件)
SQL AND & OR(逻辑与&逻辑或)
SQL ORDER BY(排序操作，BY跟排序字段)
SQL INSERT INTO VALUES(插入数据)
SQL UPDATE SET(更新数据)
SQL DELETE FROM(删除数据)
```

## 高级

```sql
SQL LIMIT(取第N到第M条记录)
SQL IN(用于子查询)
SQL BETWEEN AND(设置区间)
SQL LIKE(匹配通配符)
SQL GROUP BY(按组查询)
SQL HAVING(跟在“GROUP BY”语句后面的设置条件语句)
SQL ALIAS(AS)(可以为表或列取别名)
SQL LEFT JOIN/RIGHT/FULL JOIN(左连接/右连接/全连接)
SQL OUT/INNER JOIN(内连接/外连接)
SQL UNION/UNION ALL(并集，后者不去重)
SQL INTERSECT(交集)
SQL EXCEPT(差集)
SQL SELECT INTO(查询结果赋给变量或表)
SQL CREATE TABLE(创建表)
SQL CREATE VIEW AS(创建视图)
SQL CREATE INDEX(创建索引)
SQL CREATE PROCEDURE BEGIN END(创建存储过程)
SQL CREATE TRIGGER T_name BEFORE/AFTER INSERT/UPDATE/DELETE ON MyTable FOR (创建触发器)
SQL ALTER TABLE ADD/MODIFY COLUMN/DROP(修改表:增加字段/修改字段属性/删除字段)
SQL UNIQUE(字段、索引的唯一性约束)
SQL NOT NULL(定义字段值非空)
SQL AUTO_INCREMENT(字段定义为自动添加类型)
SQL PRIMARY KEY(字段定义为主键)
SQL FOREIGN KEY(创建外键约束)
SQL CHECK(限制字段值的范围)
SQL DROP TABLE/INDEX/VIEW/PROCEDURE/TRIGGER (删除表/索引/视图/存储过程/触发器)
SQL TRUNCATE TABLE(删除表数据，不删表结构)
```

## 函数

###常用的文本处理函数

```sql
SQL Length(str)(返回字符串str长度)
SQL Locate(substr,str)(返回子串substr在字符串str第一次出现的位置)
SQL LTrim(str)(移除字符串str左边的空格)
SQL RTrim(str)(移除字符串str右边的空格)
SQL Trim(str)(移除字符串str左右两边的空格)
SQL Left(str,n)(返回字符串str最左边的n个字符)
SQL Right(str,n)(返回字符串str最右边的n个字符)
SQL Soundex()
SQL SubString(str,pos,len)/Substr()(从pos位置开始截取str字符串中长度为的字符串)
SQL Upper(str)/Ucase(str)(小写转化为大写)
SQL Lower(str)/Lcase(str)(大写转化为小写)
```

### 常用的日期与时间处理函数

```sql
SQL AddDate()(增加一个日期，天、周等)
SQL AddTime()(增加一个时间，天、周等)
SQL CurDate()(返回当前日期)
SQL CurTime()(返回当前时间)
SQL Date()(返回日期时间的日期部分)
SQL DateDiff()(计算两个日期之差)
SQL Date_Add()(高度灵活的日期运算函数)
SQL Date_Format()(返回一个格式化的日期或时间串)
SQL Day()(返回一个日期的天数部分)
SQL DayOfWeek()(返回一个日期对应的星期几)
SQL Hour()(返回一个时间的小时部分)
SQL Minute()(返回一个时间的分钟部分)
SQL Month()(返回一个日期的月份部分)
SQL Now()(返回当前日期和时间)
SQL Second()(返回一个时间的秒部分)
SQL Time()(返回一个日期时间的时间部分)
SQL Year()(返回一个日期的年份部分)
```

### 常用的数值处理函数

```sql
SQL Avg()(求均值)
SQL Max()(求最大值)
SQL Min()(求最小值)
SQL Sum()(求和)
SQL Count()(统计个数)
SQL Abs()(求绝对值)
SQL Cos()(求一个角度的余弦值)
SQL Exp(n)(求e^n)
SQL Mod()(求余)
SQL Pi()(求圆周率)
SQL Rand()(返回一个随机数)
SQL Sin()(求一个角度的正弦值)
SQL Sqrt()(求一个数的开方)
SQL Tan()(求一个角度的正切值)
SQL Mid(ColumnName,Start,[,length])(得到字符串的一部分)
SQL Round(n,m)(以m位小数来对n四舍五入)
SQL Convert(xxx,TYPE)/Cast(xxx AS TYPE) (把xxx转为TYPE类型的数据)
SQL Format() (用来格式化数值)
SQL First(ColumnName)(返回指定字段中第一条记录)
SQL Last(ColumnName)(返回指定字段中最后一条记录)
```

## 优化技巧

https://zhuanlan.zhihu.com/p/27540896

1、应尽量避免在 where 子句中使用!=或<>操作符，否则将引擎放弃使用索引而进行全表扫描。

2、对查询进行优化，应尽量避免全表扫描，首先应考虑在 where 及 order by 涉及的列上建立索引。

3、应尽量避免在 where 子句中对字段进行 null 值判断，否则将导致引擎放弃使用索引而进行全表扫描，如：

select id from t where num is null

可以在num上设置默认值0，确保表中num列没有null值，然后这样查询：

select id from t where num=0

4、尽量避免在 where 子句中使用 or 来连接条件，否则将导致引擎放弃使用索引而进行全表扫描，如：

select id from t where num=10 or num=20

可以这样查询：

select id from t where num=10

union all

select id from t where num=20

5、下面的查询也将导致全表扫描：(不能前置百分号)

select id from t where name like ‘%c%’

若要提高效率，可以考虑全文检索。

6、in 和 not in 也要慎用，否则会导致全表扫描，如：

select id from t where num in(1,2,3)

对于连续的数值，能用 between 就不要用 in 了：

select id from t where num between 1 and 3

7、如果在 where 子句中使用参数，也会导致全表扫描。因为SQL只有在运行时才会解析局部变量，但优化程序不能将访问计划的选择推迟到运行时；它必须在编译时进行选择。然 而，如果在编译时建立访问计划，变量的值还是未知的，因而无法作为索引选择的输入项。如下面语句将进行全表扫描：

select id from t where num=@num

可以改为强制查询使用索引：

select id from t with(index(索引名)) where num=@num

8、应尽量避免在 where 子句中对字段进行表达式操作，这将导致引擎放弃使用索引而进行全表扫描。如：

select id from t where num/2=100

应改为:

select id from t where num=100*2

9、应尽量避免在where子句中对字段进行函数操作，这将导致引擎放弃使用索引而进行全表扫描。如：

select id from t where substring(name,1,3)=’abc’–name以abc开头的id

select id from t where datediff(day,createdate,’2005-11-30′)=0–’2005-11-30′生成的id

应改为:

select id from t where name like ‘abc%’

select id from t where createdate>=’2005-11-30′ and createdate<’2005-12-1′

10、不要在 where 子句中的“=”左边进行函数、算术运算或其他表达式运算，否则系统将可能无法正确使用索引。

11、在使用索引字段作为条件时，如果该索引是复合索引，那么必须使用到该索引中的第一个字段作为条件时才能保证系统使用该索引，否则该索引将不会被使 用，并且应尽可能的让字段顺序与索引顺序相一致。

12、不要写一些没有意义的查询，如需要生成一个空表结构：

select col1,col2 into #t from t where 1=0

这类代码不会返回任何结果集，但是会消耗系统资源的，应改成这样：

create table #t(…)

13、很多时候用 exists 代替 in 是一个好的选择：

select num from a where num in(select num from b)

用下面的语句替换：

select num from a where exists(select 1 from b where num=a.num)

14、并不是所有索引对查询都有效，[SQL](https://link.zhihu.com/?target=http%3A//cda.pinggu.org/view/22577.html)是根据表中数据来进行查询优化的，当索引列有大量数据重复时，SQL查询可能不会去利用索引，如一表中有字段 sex，male、female几乎各一半，那么即使在sex上建了索引也对查询效率起不了作用。

15、索引并不是越多越好，索引固然可以提高相应的 select 的效率，但同时也降低了 insert 及 update 的效率，因为 insert 或 update 时有可能会重建索引，所以怎样建索引需要慎重考虑，视具体情况而定。一个表的索引数最好不要超过6个，若太多则应考虑一些不常使用到的列上建的索引是否有 必要。

16.应尽可能的避免更新 clustered 索引数据列，因为 clustered 索引数据列的顺序就是表记录的物理存储顺序，一旦该列值改变将导致整个表记录的顺序的调整，会耗费相当大的资源。若应用系统需要频繁更新 clustered 索引数据列，那么需要考虑是否应将该索引建为 clustered 索引。

17、尽量使用数字型字段，若只含数值信息的字段尽量不要设计为字符型，这会降低查询和连接的性能，并会增加存储开销。这是因为引擎在处理查询和连接时会 逐个比较字符串中每一个字符，而对于数字型而言只需要比较一次就够了。

18、尽可能的使用 varchar/nvarchar 代替 char/nchar ，因为首先变长字段存储空间小，可以节省存储空间，其次对于查询来说，在一个相对较小的字段内搜索效率显然要高些。

19、任何地方都不要使用 select * from t ，用具体的字段列表代替“*”，不要返回用不到的任何字段。

20、尽量使用表变量来代替临时表。如果表变量包含大量数据，请注意索引非常有限（只有主键索引）。

21、避免频繁创建和删除临时表，以减少系统表资源的消耗。

22、临时表并不是不可使用，适当地使用它们可以使某些例程更有效，例如，当需要重复引用大型表或常用表中的某个数据集时。但是，对于一次性事件，最好使 用导出表。

23、在新建临时表时，如果一次性插入数据量很大，那么可以使用 select into 代替 create table，避免造成大量 log ，以提高速度；如果数据量不大，为了缓和系统表的资源，应先create table，然后insert。

24、如果使用到了临时表，在存储过程的最后务必将所有的临时表显式删除，先 truncate table ，然后 drop table ，这样可以避免系统表的较长时间锁定。

25、尽量避免使用游标，因为游标的效率较差，如果游标操作的数据超过1万行，那么就应该考虑改写。

26、使用基于游标的方法或临时表方法之前，应先寻找基于集的解决方案来解决问题，基于集的方法通常更有效。

27、与临时表一样，游标并不是不可使用。对小型数据集使用 FAST_FORWARD 游标通常要优于其他逐行处理方法，尤其是在必须引用几个表才能获得所需的数据时。在结果集中包括“合计”的例程通常要比使用游标执行的速度快。如果开发时 间允许，基于游标的方法和基于集的方法都可以尝试一下，看哪一种方法的效果更好。

28、在所有的存储过程和触发器的开始处设置 SET NOCOUNT ON ，在结束时设置 SET NOCOUNT OFF 。无需在执行存储过程和触发器的每个语句后向客户端发送 DONE_IN_PROC 消息。

29、尽量避免向客户端返回大数据量，若[数据](https://link.zhihu.com/?target=http%3A//cda.pinggu.org/)量过大，应该考虑相应需求是否合理。

30、尽量避免大事务操作，提高系统并发能力。

##操作

**条件选择 and， or，in**

```
select * from DataAnalyst
where (city = '上海' and positionName = '数据分析师') 
   or (city = '北京' and positionName = '数据产品经理')
```

```
select * from DataAnalyst
where city in ('北京','上海','广州','深圳','南京')
```

**区间数值，between and**

```
select * from DataAnalyst
where companyId between 10000 and 20000
```

between and 包括数值两端的边界，等同于 companyId >=10000 and companyId <= 20000。 

**模糊查找，like**

```
select * from DataAnalyst
where positionName like '%数据分析%'
```

where name like ’A%’ or ’B%’（错误），where name like('A%' or 'B%' )（错误），

WHERE name LIKE 'A%' OR name LIKE 'B%';（正确）。

**%代表的是通配符 **

**not，代表逻辑的逆转，常见not in、not like、not null等。 **

in的对立面并不是NOT IN！not in等价的含义是<> all，例如In(‘A’,’B’)：A或者B；not in (‘A’,’B’)：不是A且B。 

**group by**

```
select city,count(1) from DataAnalyst
group by city
# 去重
select city,count(distinct positionId) from DataAnalyst
group by city
# 多维度
select city,workYear,count(distinct positionId) from DataAnalyst
group by city,workYear
```

上述语句，使用count函数，统计计数了每个城市拥有的职位数量。括号里面的1代表以第一列为计数标准。 

除了count，还有max，min，sum，avg等函数，也叫做聚合函数。 

**逻辑判断**

统计各个城市中有多少数据分析职位，其中，电商领域的职位有多少，在其中的占比？ 

```
select if(industryField like '%电子商务%',1,0) from DataAnalyst
```

利用if判断出哪些是电商行业的数据分析师，哪些不是。if函数中间的字段代表为true时返回的值，不过因为包含重复数据，我们需要将其改成positionId。图片中第二个count我漏加distinct了。之后，用它与group by 组合就能达成目的了。 

```
select city,
       count(distinct positionId),
       count(distinct if(industryField like '%电子商务%',positionId,null)) 
from DataAnalyst
group by city
```

第一列数字是职位总数，第二列是电商领域的职位数，相除就是占比。记住，**count是不论0还是1都会纳入计数，所以第三个参数需要写成null**，代表不是电商的职位就排除在计算之外。 

找出各个城市，数据分析师岗位数量在500以上的城市有哪些，应该怎么计算？有两种方法，第一种，是使用having语句，它对聚合后的数据结果进行过滤。 

```
select city,count(distinct positionId) from DataAnalyst
group by city having count(distinct positionId) >= 500 
```

第二种，是利用嵌套子查询 

```
select * from(
    select city,count(distinct positionId) as counts from DataAnalyst
    group by city) as t1
where counts>=500
```

**时间**

```
select now()    # 获得当前的系统时间，精确到秒
select date(now())
# 它代表的是获得当前日期，week函数获得当前第几周，month函数获得当前第几个月。其余还包括，quarter，year，day，hour，minute。
select week(now(),0)
# 除了以上的日期表达，也可以使用dayofyear、weekofyear 的形式计算。
```

```
# 时间的加减
select date_add(date(now()) ,interval 1 day)
```

我们可以改变1为负数，达到减法的目的，也能更改day为week、year等，进行其他时间间隔的运算。如果是求两个时间的间隔，则是datediff(date1,date2)或者timediff(time1,time2)。 

**数据清洗类**

```
select left(salary,1) from DataAnalyst
```

MySQL支持left、right、mid等函数，和Excel一样。 

首先利用locate函数查找第一个k所在的位置。 

```
select locate("k",salary),salary from DataAnalyst
```

然后使用left函数截取薪水的下限。

```
select left(salary,locate("k",salary)-1),salary from DataAnalyst
```

为了获得薪水的上限，要用substr函数，或者mid，两者等价。

> substr（字符串，从哪里开始截，截取的长度）

再然后计算不同城市不同工作年限的平均薪资。 

```
select city,workYear,avg((bottomSalary+topSalary)/2) as avgSalary
from (select left(salary,locate("K",salary)-1) as bottomSalary,
             substr(salary,locate("-",salary)+1,length(salary)- locate("-",salary)-1) as topSalary,
             city,positionId,workYear
      from DataAnalyst
      where salary not like '%以上%') as t1
group by city,workYear
order by city,avgSalary 
```

一些雷区要注意：

①在不用聚合函数的时候，单独用group by，group by 子句中必须包含所有的列，否则会报错，但此时虽然成功执行了，group by在这里并没有发挥任何的作用，完全可以不用；若不用聚合函数，就是按照group by后面字段的顺序，把相同内容归纳在一起

③如果只有聚合函数，而没有group by，则聚合函数用于聚合整个结果集 (匹配WHERE子句的所有行)，相当于只分一组。

④where后面不能放聚合函数！无论是count还是sum。那么如何解决呢，使用HAVING关键字！例如：having
sum(amount) >100

⑤order by 后面是可以跟聚合函数的，即可以用聚合函数排序。

另外，除了Count(*)函数外，所有的聚合函数都忽略NULL值。

两个典型小问题的解决方法，看了很受启发。一是，最后排序时若要将某一类放在最前或最后，可以利用case when，巧妙的引用辅助列，帮助排序。例如：

①ORDER BY (case
when subject in ('Physics','Chemistry') then 1 else 0 end ), subject, winner

结果：科目为(‘Physics’,’Chemistry’)
的排在最后，其余科目按subject升序排列，

②ORDER BY (case
when subject in ('Physics','Chemistry') then 1 else 0 end ) desc, yr desc, winner

结果：将(‘Physics’,’Chemistry’)
排在最前；同一科目种类时，按年份从新到老；同一科目、同一年份时，按获奖者名字升序排列。

二是，一个经典问题：分组后取每组的前几条记录。这里看一个例子吧。

例：已知一个表， StudentGrade
(stuid--学号, subid--课程号, grade--成绩)。PRIMARY KEY
(stuid, subid)。

想要：查询每门课程的前2名成绩。

方法①：

select distinct * from
studentgrade as t1

where stuid in

(select top 2 stuid from
studentgrade as t2

where t1.subid=t2.subid

order by t2.grade desc) order by subid, grade
desc

思路：相同的表格自联结，第二个表格将相同学科的所有学生按成绩排序-倒序，选取前二。注意，mysql不支持select top n的语法！但是mysql可用limit来实现相关功能。

方法②：

select * from StudentGrade a

where (select count(1) from
studentGrade b

where b.subId=a.subId and b.grade
\>= a.grade) <=2

思路：第一个>=号，限制了查询条件是相同科目下成绩从大往小排，第二个<=号，表示筛选个数是2个（从1开始的）。

注意，这里大于等于、小于等于容易弄错，尤其是第二个。

方法③：

select * from StudentGrade a

where (select count(1) from
StudentGrade b

where b.subid=a.subid and
b.grade> a.grade) <=1

order by subId, grade desc

思路：这两张表思路相同：相同表格自联结，返回相同学科并且成绩大于a表的影响行数。这就是查询条件，再按 subId,grade 排序。