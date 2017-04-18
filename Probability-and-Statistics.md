
## 关于概率的一些定义
### 公理
* 公理一 对于每一个事件（Event）A，它的概率 Pr(A) >= 0
* 公理二 整个样本空间事件的概率为1，Pr(S) = 1
* 公理三 对于每一个无限不相关事件序列：A1, A2, ..., 事件A = A1 ∪ A2 ∪ ..., 那么 Pr(A) = sum\<i = 1 to infinity>(Pr(Ai))

### 基本定理
* 定理一 空集概率为0  **Pr(∅) = 0**
* 定理二 对于每一个有限不相关事件序列：A1，A2，...，An, 事件A = A1 ∪ A2 ∪ ... ∪ An， 那么 **Pr(A) = sum\<i = 1 to n>(Pr(Ai))**
* 定理三 对于任意事件A  **Pr(A^c) = 1 - Pr(**, 其中 A^c 为 A 的补集
* 定理四 如果 A ⊂ B，那么 **Pr(A) ⊂ Pr(B)**
* 定理五 对于任意事件A  **0 <= Pr(A) <= 1**
* 定理六 对于任意事件A，B  **Pr(A ∩ B^c) = Pr(A) - Pr(A ∩ B)**
* 定理七 对于任意事件A，B  **Pr(A ∪ B) = Pr(A) + Pr(B) − Pr(A ∩ B)**
* 定理八 Bonferroni inequality, 对于所有事件A1, A2, ..., An, A_Intersection = A1 ∩ A2 ∩ ... ∩ An, A_Union = A1 ∪ A2 ∪ ... ∪ An
        **Pr(A_Union) <= sum\<i = 1 to n>(Pr(Ai)**
        **Pr(A_Intersection) >= 1 - sum\<i = 1 to n>(Pr(Ai^c)**
