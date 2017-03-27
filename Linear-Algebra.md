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
- 一组向量的张成另一组向量，其中每个向量都是原向量组的线性组合
- 如果一组向量 {x1, x2, ..., xn} (x_i ∈ R^n) 是线性无关的，那么它的张成 span({x1, x2, ..., xn}) = R^n
  - 一组线性无关向量的张成是向量维度的整个空间，该n维空间中的任意一个向量都可以表示成这组向量的线性组合
#### Projection
- 一个向量 y ∈ R^m 在span({x1, x2, ..., xn}) [xi ∈ R^m] 中的投影就是此张成空间中距离向量y最近的向量v
  - v是y在span({x1, x2, ..., xn})中的投影，那么 v ∈ span({x1, x2, ..., xn}), 且使||v - y||\_2最小
  - Proj(y;{x1, x2, ..., xn} = argmin[v∈span({x1, x2, ..., xn})]||y - v||\_2
#### Range
- 矩阵A ∈ R^m\*n 的范围 R(A)，又叫列空间，是矩阵A的所有列向量的张成
  - R(A) = {v ∈ R^m : v = Ax, x ∈ R^n}
  - 就是矩阵所能表示的空间中所有向量的集合
#### Nullspace
- 矩阵 A ∈ R^m\*n 的零空间 N(A), 是所有与矩阵A相乘以后为0的向量集合
  - N(A) = {x ∈ R^n : Ax = 0}
- 对于 A ∈ R^m\*n, R(A)中的向量是m维，而 N(A)中的向量是n维，R(A^T)也是n维的
- R(A^T)和N(A)是两个不相交向量集，它们一起充满了整个R^n空间（n维空间）
  - 这种类型的向量集合叫正交补集，R(A^T) = N(A)^⊥

### Determinant
- 方阵 A ∈ R^n\*n 的行列式是一个映射函数 det : R^n\*n -> R, 记为|A|或者detA
- 行列式从几何角度看就是矩阵各向量（以及对应平行向量）围成的空间的大小
  - 二维矩阵表示平面，它的行列式就是矩阵围成的平行四边形的面积
  - 三维矩阵是空间，行列式是矩阵围成空间的平行六面体的体积
- 行列式具有以下性质
  - 单位矩阵的行列式为1，|I| = 1
  - 对于矩阵 A ∈ R^m\*n，如果其中某个行向量乘以一个实数 t ∈ R, 则新的行列式变为原来的t倍，t|A|
    - 如果二维矩阵其中一个行向量长度变成原来的t倍，那面积也会相应变化t倍
  - 交换矩阵中任意两个行向量的位置，新的矩阵的行列式变为 -|A|
  - For A ∈ R^n\*n, |A| = |A^T|
  - For A, B ∈ R^n\*n, |AB| = |A||B|
  - For A ∈ R^n\*n, 只有当A为非可逆矩阵时，|A| = 0
    - 非可逆矩阵一般不是满秩的，说明被降维，比如只能表示平面中的一条线（两个向量位于同一条线上），他们围成的面积为0
  - For A ∈ R^n\*n，并且A为可逆矩阵，那么|A^-1| = 1/|A|
- 对于矩阵 A ∈ R^n\*n, 定义 A\_(\i,\j) ∈ R^(n-1)\*(n-1), 这是从A删掉第i行和第j列之后的新矩阵，那么行列式的一般定义为
  - |A| = sum{i=1 - n}(-1)^(i+j) * a_ij * |A\_(\i,\j)|,   for any j ∈ 1, ..., n
        = sum{j=1 - n}(-1)^(i+j) * a_ij * |A\_(\i,\j)|,   for any i ∈ 1, ..., n
    for A ∈ R^1\*1, |A| = a_11
#### classical adjoint
- 方阵 A ∈ R^n\*n 的古典伴随记为 adj(A)
  - adj(A) ∈ R^n\*n, (adj(A))\_ij = (-1)^(i+j) * |A\_(\j,\i)|
  - 对于任意的非奇异矩阵（可逆矩阵）A ∈ R^n\*n
    - A^-1 = (1/|A|) * adj(A)
    - 可以看成，使用递归算法可以很方便的算出A的逆矩阵 

### Quadratic Forms and Positive Semidefinite Matrices
- 对于方阵 A ∈ R^n\*n 和向量 x ∈ R^n，标量值 x^T \* A \* x 叫做二次式，根据向量计算可得出公式
  - x^T \* A \* x = sum{i=1,n}(sum{j=1,n}(A_ij * x_i * x_j))
    - 此等式说明一个标量的转置就是它自己
  - x^T \* A \* x = x^T(1/2 * A + 1/2 * A^T)x
    - 此等式说明在给两个相等值取平均？？？
  - 只有矩阵A的对称部分参与了二次式计算
    - 通常假设在二次式中出现的矩阵都是对称矩阵
- 给出以下定义
  - 对称矩阵 A ∈ S^n 是正定的(Positive Definite)，如果对于所有非零向量 x ∈ R^n, 它的二次式为正, x^T * A * x > 0
    - 通常记为 A > 0，用S^n\_++表示所有正定矩阵
  - 对称矩阵 A ∈ S^n 是正半定的(Positive Semidefinite)，如果对于所有向量，它的二次式都为非负， x^T * A * x >= 0
    - 记为 A >= 0, 用S^n\_+表示所有这种矩阵
  - 对称矩阵 A ∈ S^n 是负定的(Negative Definite)，如果所有非零向量 x ∈ R^n, 它的二次式为负，x^T * A * x < 0
    - 记为 A < 0
  - 对称矩阵 A ∈ S^n 是负半定的(Negative Semidefinite)，如果对所有向量 x ∈ R^n，它的二次式不为正，x^T * A * x <= 0
    - 记为 A <= 0
  - 对称矩阵 A ∈ S^n 是不定的，如果它既不是正半定也不是负半定的
    - 比如对于 x1, x2 ∈ R^n, x1^T * A * x1 > 0, x2^T * A * x2 < 0
- 性质
  - 如果A是正定的，那么-A就是负定的，反之亦然
  - 如果A是正半定的，那么-A就是负半定的，反之亦然
  - 如果A是不定的，那么-A也是不定的
  - 正定和负定矩阵都是满秩的，并且都是可逆的
- Gram Matrix
  - 对于矩阵 A ∈ R^m\*n (不必对称也不必是方阵), 矩阵 G = A^T * A，这个矩阵叫Gram Matrix
    - Gram矩阵总是正半定的, G >= 0
    - 如果m >= n(假设A满秩), 那么Gram矩阵是正定的

### Eigenvalues and Eigenvectors
- 对于方阵 A ∈ R^n\*n, λ ∈ C (Complex numbers, 复数), x ∈ C^n, 如果满足如下条件，我们就说λ是A的一个特征值，而x是A对应的特征向量
  - Ax = λx, x != 0
  - 几何意义上看，就是向量x经过矩阵A的空间变换后，保持方向不变，只是在原向量基础上乘了一个系数
  - 对于一个特征向量x ∈ R^n, 标量 t ∈ C, tx也是一个特征向量，而我们说特征向量时，一般特指长度为1的那个（然而也有1和-1之分）
- 特征值特指向量的另一种表达方式
  - (λI - A)x = 0, x != 0
  - 我们说(λ, x)是A的特征值-向量对
  - 只有当矩阵(λI - A)存在零空间时（也就是不可逆时，是奇异矩阵），此方程才有非零解
    - 因此此矩阵的行列式为0： |(λI - A)| = 0
- 由于矩阵(λI - A)行列式为0，根据行列式定义，可以把此等式展开成一个很大的λ的多项式，其中λ的最大级数为n
  - 通过解这个多项式就可以求得这n个特征值(λ1, λ2, ..., λn)
  - 一旦求得特征值，通过解以下线性方程就可得到λ_i对应的特征向量
    - (λ_i * I - A)x = 0
- 性质，方阵 A ∈ R^n\*n 的特征值λ1, ... λi, ..., λn和对应特征向量x1, ..., xn 
  - 矩阵A的迹等于它的特征值的和
    - trA = sum{i=1,n}(λ\_i)
  - 矩阵A的行列式等于它的特征值的乘积
    - |A| = λ1 * λi * ... * λn
  - 矩阵A的秩（Rank)与A的非零特征值的数量相同
  - 如果A是非奇异矩阵（可逆的），那么1/λi 是矩阵A^-1 与特征向量xi对应的特征值
  - 对角矩阵 D = diag(d1, ..., dn)的特征值就是对角上的元素值
- 如果 Λ 是一个对角矩阵，其值为 矩阵 A ∈ R^n\*n的特征值，那么我们可以把特征向量（列向量）也组成一个矩阵 X ∈ R^n\*n
  - AX = XΛ
  - 如果矩阵A的特征向量集合是线性无关的，那么矩阵X就是可逆矩阵，则有
    - A = XΛX^-1
    - 可以写成这种形式的矩阵就是可对角化的(diagnonalizable)
    
### EigenValues and Eigenvectors of Symmetric Matrices
a

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
### Inverse Matrix
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
