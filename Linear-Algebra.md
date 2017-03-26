# Linear Algebra
线性代数提供了一种简洁的表述和计算线性方程组的方法。比如下面的一个二元一次方程组，可以用矩阵求解
```
x + 2y = 3
3x - 5y = 7
- 矩阵表示 Ax = b
  * A是一个矩阵，由未知数的参数组成
  * x是未知数组成的二维列向量
  * b是等式右边的实数组成的二维列向量
1  2   x    3
3 -5   y =  7
```

## Basic Notation
- R^m*n(m*n应该上标，但是github markdown貌似不支持上标语法)表示m行，n列矩阵
- A ∈ R^m*n, A是一个m行n列矩阵，其中的元素都是实数
- x ∈ R^n(n为上标), x是一个n维列向量(Column Vector)，可以看成一个n*1矩阵。x也可以是一个n维行向量(Row Vector)，行向量一般用x^T（T为上标）表示，可以看成一个1*n矩阵。如果元素相同，x^T与x互为转置矩阵（Transpose Matrix）。
- x_i（i为下标）表示向量的第i个元素
- a_ij(ij为下标)表示矩阵A位于第i行，第j列的元素
- a_j(j为下标)表示矩阵A的第j列向量
- a_i^T(i为下标，T为上标)表示矩阵A的第i行向量

## Matrix Multiplication
```
矩阵A ∈ R^m*n，B ∈ R^n*p, 他们相乘的结果为C ∈ R^m*p,是一个m行p列矩阵
       C = AB ∈ R^m*pa
       C_ij = sum{k=1}^n(A_ik * Bkj)  [sum{k=1}^n就是k为1到n时，参数的和]
从公式中可以看成，乘法的前一个矩阵的列数与后一矩阵的行数要相等。
```
### Vector-Vector Products
#### Row Vector - Column Vector Product
- Inner Product or Dot Product, 又叫内积或者点积，结果为一个实数a
- x, y ∈ R^n, x^Ty ∈ R
- 行向量与列向量相乘结果是一个实数, 值为行列向量对应元素乘积的和: x^Ty = sum{i=1}^n(x_i*y_i) 
- 从几何角度看，如果x，y为二维向量，点积就是这两个向量以及他们的平行向量围成的平行四边形面积，如果是三维向量则为体积

#### Column Vector - Row Vector Product
- Outer Product, 外积a
- x ∈ R^m, y ∈ R^n, xy^T ∈ R^m*n，外积结果为m*n矩阵
- 结果矩阵中每个元素的值为对应列向量元素与相应行向量元素的乘积：(xy^T)_ij = x_i*y_j

### Matrix-Vector Products
#### Matrix - Column Vector Product
- A ∈ R^m*n, x ∈ R^n, y = Ax = a_i^T * x ∈ R^m, 结果为一个m维列向量
- 如果把A看成由m个行向量组成的矩阵，那么 Ax 的第i个元素就是A中第i个行向量与x向量的内积
- 如果把A看成由n个列向量组成的矩阵，那么 Ax 就是由A的各列向量相加组成，其中第i列的系数是x向量的第i个元素x_i: a1*x1 + a2*x2 + ... + an*xn
  - 换句话说，y是A的列向量的**线性组合 Linear Combination**，其系数由x提供
 
#### Row Vector - Matrix product
- x ∈ R^m, A ∈ R^m*n, y^T = x^T * A, y ∈ R^n，结果为一个n维行向量
- 如果把A看成由n个列向量组成的矩阵，那么x^T * A的第i个元素就是x^T与A中第i个列向量的内积：[x^T * a1   x^T * a2   ...   x^T * a_n]
- 如果把A看成由m个行向量组成的矩阵，那么x^T * A的结果就是A的行向量的线性组合，其系数是x向量中对应的元素

### Matrix-Matrix Proudcts
#### View Matrix-Matrix Products as Vector-Vector Products
- 通过对向量与向量以及向量与矩阵乘积的分析，我们可以把矩阵的乘法看成是矩阵中向量之间的乘法
  - 从定义就可看出，矩阵乘积的元素(AB)_ij是A的第i个行向量与B的第i个列向量的内积，所以(AB)_ij = a_i^T * b_i
- 如果把A看成由n个列向量组成的行向量，B看出由n个行向量组成的列向量，AB的结果就可以看出是所有列向量与对应行向量的外积的和

#### View Matrix-Matrix Products as Matrix-Vector Products
- 如果把B看成由p个列向量组成，那AB的第i列可看成是A与B中第i个列向量相乘的结果： c_i = Abi
- 如果把A看成由m个行向量组成，那么AB的第i行可看成是A的第i个行向量与B的乘积: c_i^T = a_i^T * B

#### Properties of Matrix Multiplication
- 结合律：(AB)C = A(BC)
- 分配率：A(B + C) = AB + AC
- 不满足交换律：AB != BA

## Operations and Properties
### Identity Matrix and Diagonal Matrices
#### Diagonal Matrices
- 除对角线外，其他元素都为0的方阵，叫对角矩阵
- 对角矩阵表示为 D = diag(d1, d2, ..., dn), where Dii = di

#### Identity Matrix
- 单位矩阵是对角线元素为1，其他元素为0的方阵，用I表示，I ∈ R^n*n
  - AI = A = IA
    - for A ∈ R^m\*n, I_n ∈ R^n\*n, I_m ∈ R^m\*m:  AI_n = A, A=I_mA
- 单位矩阵是特殊的对角矩阵: I = diag(1, 1, ..., 1)

### The Transpose
- 矩阵A的转置是沿A的对角线翻转之后的矩阵，A ∈ R^m*\n, A^T ∈ R^n\*m
- 转置矩阵的性质
  - (A^T)^T = A
  - (AB)^T = B^T*A^T
  - (A + B)^T = A^T + B^T
 
### Symetric Matrices
- A ∈ R^n*n
  - 如果 A = A^T，那么A是一个对称矩阵
  - 如果 A = -A^T，那么A是一个反对称矩阵
  - 矩阵 A + A^T 一定是一个对称矩阵
  - 而 A - A^T 是一个反对称矩阵
  - 任何一个方阵都可以用对称和反对称矩阵来表示
    - A = 1/2(A + A^T) + 1/2(A - A^T)
- S^n来表示n维对称矩阵合集，A ∈ S^n，则A为n维对称矩阵

### The Trace
- 只有**方阵**才有迹(Trace), A ∈ R^n*n，A的迹用 tr(A)，或者trA表示，值为A所有对角线元素的和
  - trA = sum{i=1 - n}(A_ii)
- 迹的性质
  - A ∈ R^n\*n, trA = tr(A^T)
  - A, B ∈ R^n\*n, tr(A + B) = trA + trB
  - A ∈ R^n\*n, t ∈ R, tr(tA) = t\*trA
  - For A, B, AB ∈ R^n\*n, trAB = trBA
  - For A, B, C, ABC ∈ R^n\*n, trABC = trBCA = trCAB

### Norms
- 向量的范数可以用来描述向量的长度
- l_2 范数(||x||_2)又叫欧几里得范数，其值为向量各元素的平方和再开方
  - 几何意义上，二维或三维空间中向量的长度就是它的l_2范数a
  - (||x||_2)^2 = x^T \* x, 向量x的l_2范数的平方等于x的转置向量与x的乘积
- 实际上范数可以看成一个映射函数，它把向量映射成一个实数，映射函数满足以下条件
  - For x ∈ R&n, f(x) >= 0 (非负性)
  - 只有当x = 0时，f(x) = 0 (确定性)
  - For all x ∈ R^n, t ∈ R, f(tx) = |t|f(x) (齐次性)
  - For all x, y ∈ R^n, f(x + y) <= f(x) + f(y) (三角不等式)
    - 想象平面中的三角形，任意一边的长度都不大于另两边长度之和
- 其他范数还有l_1范数，无穷范数等，他们共同定义如下(p>=1):
  - ||x||_p = (sum{i=1 -n}(|x_i|^p))^(1/p)
- 矩阵也有范数, 如Frobenius范数为 tr(A^T \* A) 的开方

### Linear Indepdence and Rank
#### Linear Independence
- 对于一组向量{x1, x2, ..., xn} ⊂ R^m, 如果里面任意一个向量都不能表示为剩余向量的线性组合(Linear Combination)，那么就说它们是线性无关的
- 相反，如果存在一个向量可以用剩余的向量的线性组合表示，这个向量组就是线性相关的
- 从几何角度看，线性相关一般意味着存在平行向量或者0向量

#### Rank
- 一个矩阵A ∈ R^m\*n 的列秩(Column Rank)，就是该矩阵所有的列向量能组成的最大线性无关向量组大小
  - 比如一个3\*3矩阵，如果其中一列可以用另两列的线性组合表示（而剩下的两列不能相互线性表示），那它的列秩就是2
- 同样的，矩阵A ∈ R^m\*n 的行秩(Row Rank)，就是所有最大的线性无关行向量集合的大小
- 矩阵的列秩和行秩相等，都可以用来作为矩阵的秩：rank(A)
- 矩阵的秩有如下性质：
  - For A ∈ R^m\*n, rank(A) <= min(m, n), if rank(A) = min(m, n), 那么就说A是满秩
  - For A ∈ R^m\*n, rank(A) = rank(A^T)
  - For A ∈ R^m\*n, B ∈ R^n\*p, rank(AB) <= min(rank(A), rank(B))
  - For A, B ∈ R^m\*n, rank(A + B) <= rank(A) + rank(B)
- 几何意义来看，矩阵的秩表示该矩阵能扩展成的空间的维度，比如一个3\*3矩阵，秩为2，那么说明有一个向量可以通过另两个向量线性组合来表示，几何上则表明这3个向量落在同一平面，他们只能表示该平面内的向量

### The Inverse
- 只有方阵A ∈ R^n\*n才有逆矩阵，但并非所有方阵都有逆矩阵（其中一个条件是满秩矩阵）
- 矩阵A的逆矩阵表示为A^(-1)
- 矩阵与其逆矩阵相乘的结果为单位矩阵A^(-1) \* A = I = A \* A^(-1)
- 如果A的逆矩阵存在，就说A是可逆的(invertible)，或者是非奇异的(non-singular)
- 如果A不存在逆矩阵，就说A是不可逆的(non-invertible)，或者奇异的(singular)
- 逆矩阵的性质(A, B ∈ R^n\*n)
  - (A^-1)^-1 = A 矩阵与其逆矩阵互逆
  - (AB)^-1 = B^-1 \* A^-1
  - (A^-1)^T = (A^T)^-1, 写成A^(-T)
- 逆矩阵可以用于求解方程组，Ax = b，那么x = A^-1 \* b，解后面的矩阵-向量乘法就可得到x的解

### Orthogonal Matrices
- 向量x, y ∈ R^n, 如果x^T \* y = 0, 那么就说x和y正交
- 向量x ∈ R^n, 如果x的l_2范数 ||x||_2 = 1，就说x是标准化向量
- 对于方阵 U ∈ R^n\*n, 如果所有列向量相互正交，并且都是标准化向量，就说矩阵U是正交矩阵
  - 正交矩阵的列矩阵就说标准正交的
  - U^T \* U = I = U \* U^T
  - 通过上一个性质可以得出正交矩阵U的转置矩阵就是它的逆矩阵：U^-1 = U^T
  - 正交矩阵与向量相乘不会改变向量的l_2范数, x ∈ R^n, U ∈ R^n\*n
    - ||Ux||_2 = ||x||_2

### Range and Nullspace of a Matrix
#### Span

#### Range
#### Nullspace


## Some concepts
### Matrix
矩阵可以看成变换函数
- 向量空间可以通过矩阵进行线性变换
- 一个向量通过矩阵可以变成另一个向量
- 一个有n行m列的矩阵叫做n*m矩阵
### Vector
向量从直观来说就是一组数据，是线性代数的基础
- 物理学角度看向量是空间中带方向的点，二维向量是平面中一段带箭头的线，可在平面中自由平移
- Column Vector: 列向量，有n个元素的列向量可以看成是一个n*1矩阵
- Row Vector: 行向量，有n个元素的行向量可以看成一个1*n矩阵
 - 行向量是对应列向量的转置向量
### Linear Transformation
线性变换通过矩阵进行，向量空间变换前后保持原点不变，线之间的距离不变（始终保持平行）
### Span
空间的扩张也通过矩阵进行，矩阵可把空间中任一点变换到另一点
- 多数情况矩阵可扩张至整个向量空间
- 特殊情况下只能在一条线上进行扩展
- 最极端情况就是空间中所有点都只能变换成原点
### Base Vector: i-hat j-hat
i向量和j向量是二维向量空间中的长度为1的基础向量，i向量沿x轴方向，j向量沿y轴方向。三维空间中另一个基础向量为k向量，沿z轴方向。
### Determination
行列式，按英文原意翻译成决定式可能更合适
- 二维平面几何中，矩阵的决定式就是矩阵中两个列向量（以及与他们各自平行的向量）围成的平行四边形面积
- 三维空间中就是矩阵3个列向量围成的平行六面体体积
### Reverse Matrix
逆矩阵，线性变换（矩阵）可以把一个向量转换成空间中另一个向量，反过来另一个向量也可以通过线性变换转换成原向量。这种反向转换就是通过逆矩阵进行的。
### Column Space
### Null Space
### Dot Production
### Duality
### Cross Production
### Basis Transformation
### Eigen Vector
### Eigen Value
### Abstract Vector Space
