# DMP PI Model RC Segment Derivative Chain

本文只讨论 DMP timing 的 **PI model** 连续分支。

不考虑以下离散或非光滑情况：

- CAP / ZERO_C2 / fallback 分支切换
- LUT bin 切换
- arrival / slew winner 切换
- root finding bracket 失败
- clamp / guard condition 触发

所有导数都是固定当前分支、固定当前 topology、固定当前 winner 后的局部解析导数。

## 1. 目标

对一段 RC segment 的参数求导：

$$
\theta = R_s
$$

或：

$$
\theta = C_s
$$

记：

$$
\dot{x} = \frac{dx}{d\theta}
$$

目标输出：

$$
\dot{d}_{cell},\quad
\dot{s}_{cell},\quad
\dot{d}_{net,k},\quad
\dot{s}_{net,k}
$$

其中：

- $d_{cell}$ 是 DMP PI cell arc delay。
- $s_{cell}$ 是 DMP PI cell output slew。
- $d_{net,k}$ 是 net 到 sink $k$ 的 delay。
- $s_{net,k}$ 是 net sink slew。

如果 segment 位于当前 cell 的 output net 上，通常当前 cell input slew 对该 segment 不变：

$$
\dot{S}_{in} = 0
$$

如果做整条 timing path 的链式传播，则上游 segment 会通过上游 slew 影响下游 cell：

$$
\dot{S}_{in} \ne 0
$$

本文保留 $\dot{S}_{in}$ 项，实际使用时可以按需要置零。

## 2. RC Tree Notation

对一条 rooted RC tree：

- root 是 driver pin node。
- $v$ 是 tree node。
- $p(v)$ 是 $v$ 的 parent node。
- $u$ 是 $v$ 的 child node。

对非 root 节点 $v$：

$$
R_v = R_{p(v)\rightarrow v}
$$

$$
q_v = \text{node capacitance at } v
$$

如果 segment $s$ 对应 tree edge $p(v)\rightarrow v$：

$$
\dot{R}_v = 1
$$

否则：

$$
\dot{R}_v = 0
$$

如果 segment capacitance 按两端各一半分配到 endpoint $a,b$：

$$
\dot{q}_a = \frac{1}{2},\qquad
\dot{q}_b = \frac{1}{2}
$$

其他节点：

$$
\dot{q}_v = 0
$$

如果实际构图阶段 cap 分配不是一半一半，把 $\frac{1}{2}$ 换成实际分配权重即可。

## 3. RC Tree To DMP Moments

DMP 对每个 node 自底向上计算 reduced parasitic moments。定义：

$$
M_v \leftrightarrow y1[v]
$$

$$
N_v \leftrightarrow y2[v]
$$

$$
P_v \leftrightarrow y3[v]
$$

对节点 $v$：

$$
M_v = q_v + \sum_{u\in child(v)} M_u
$$

$$
N_v =
\sum_{u\in child(v)}
\left(
N_u - R_u M_u^2
\right)
$$

$$
P_v =
\sum_{u\in child(v)}
\left(
P_u - 2R_uM_uN_u + R_u^2M_u^3
\right)
$$

## 4. Moment Derivatives

对 $M_v$：

$$
\dot{M}_v =
\dot{q}_v
+ \sum_{u\in child(v)} \dot{M}_u
$$

对 $N_v$：

$$
\dot{N}_v =
\sum_{u\in child(v)}
\left[
\dot{N}_u
- \dot{R}_u M_u^2
- 2R_uM_u\dot{M}_u
\right]
$$

对 $P_v$：

$$
\dot{P}_v =
\sum_{u\in child(v)}
\left[
\dot{P}_u
- 2\dot{R}_uM_uN_u
- 2R_u(\dot{M}_uN_u + M_u\dot{N}_u)
+ 2R_u\dot{R}_uM_u^3
+ 3R_u^2M_u^2\dot{M}_u
\right]
$$

这一步可以自底向上做一次 derivative propagation。

## 5. RC Tree To Sink Elmore

sink node $k$ 的 Elmore delay：

$$
E_k =
\sum_{v\in path(root,k),\,v\ne root}
R_vM_v
$$

导数：

$$
\dot{E}_k =
\sum_{v\in path(root,k),\,v\ne root}
\left(
\dot{R}_vM_v + R_v\dot{M}_v
\right)
$$

因此，对路径上的 segment edge $s$：

$$
\frac{\partial E_k}{\partial R_s} = M_s
$$

如果 $s$ 不在 $root\rightarrow k$ 路径上，则 direct resistance term 为 0；但如果该 segment 的 capacitance 改变了路径上某些 downstream moment，仍然可能通过 $\dot{M}_v$ 影响 $E_k$。

## 6. Root Moments To PI Model

root moments：

$$
M = M_{root},\qquad
N = N_{root},\qquad
P = P_{root}
$$

PI model：

$$
C_1 = \frac{N^2}{P}
$$

$$
C_2 = M - C_1
$$

$$
r_\pi = -\frac{P^2}{N^3}
$$

导数：

$$
\dot{C}_1 =
\frac{2N}{P}\dot{N}
- \frac{N^2}{P^2}\dot{P}
$$

$$
\dot{C}_2 =
\dot{M} - \dot{C}_1
$$

$$
\dot{r}_\pi =
-\frac{2P}{N^3}\dot{P}
+ \frac{3P^2}{N^4}\dot{N}
$$

到这里，segment RC 的影响已经转换为：

$$
\dot{C}_1,\quad
\dot{C}_2,\quad
\dot{r}_\pi,\quad
\dot{E}_k
$$

## 7. Liberty LUT Derivatives

令：

$$
D(S,C) = \text{Liberty cell delay LUT}
$$

$$
U(S,C) = \text{Liberty output slew LUT}
$$

其中 $S$ 是 input slew，$C$ 是 load capacitance。

固定 LUT bin 内，代码使用双线性插值。因此：

$$
\dot{D} =
D_S\dot{S} + D_C\dot{C}
$$

$$
\dot{U} =
U_S\dot{S} + U_C\dot{C}
$$

$D_S,D_C,U_S,U_C$ 是当前 LUT cell 内双线性函数的局部偏导。

## 8. Driver Resistance Derivative

DMP PI 分支先用总 cap：

$$
C_{tot} = C_1 + C_2
$$

table delay：

$$
d_1 = D(S_{in}, C_{tot})
$$

代码用一个小 cap perturbation 估计 driver resistance：

$$
\Delta C = \frac{10^{-15}}{cap\_unit}
$$

$$
d_2 = D(S_{in}, C_{tot} + \Delta C)
$$

$$
r_d =
-\ln(v_{th})\,
\frac{|d_1-d_2|}{\Delta C}
$$

固定：

$$
\sigma = \operatorname{sign}(d_1-d_2)
$$

则：

$$
\dot{r}_d =
-\ln(v_{th})\,
\frac{\sigma}{\Delta C}
\left(
\dot{d}_1-\dot{d}_2
\right)
$$

展开：

$$
\dot{d}_1 =
D_S(S_{in},C_{tot})\dot{S}_{in}
+ D_C(S_{in},C_{tot})\dot{C}_{tot}
$$

$$
\dot{d}_2 =
D_S(S_{in},C_{tot}+\Delta C)\dot{S}_{in}
+ D_C(S_{in},C_{tot}+\Delta C)\dot{C}_{tot}
$$

$$
\dot{C}_{tot} =
\dot{C}_1+\dot{C}_2
$$

## 9. PI Coefficients

DMP PI response 使用：

$$
z_1 = \frac{1}{r_\pi C_1}
$$

$$
k_0 = \frac{1}{r_d C_2}
$$

$$
a = r_\pi r_d C_1 C_2
$$

$$
b = r_d(C_1+C_2) + r_\pi C_1
$$

$$
\Delta = b^2 - 4a
$$

$$
g = \sqrt{\Delta}
$$

$$
p_1 = \frac{b+g}{2a}
$$

$$
p_2 = \frac{b-g}{2a}
$$

$$
k_2 = \frac{z_1}{p_1p_2}
$$

$$
k_1 =
\frac{1-k_2(p_1+p_2)}
{p_1p_2}
$$

$$
k_4 =
\frac{k_1p_1+k_2}
{p_2-p_1}
$$

$$
k_3 = -k_1-k_4
$$

另有：

$$
z =
\frac{C_1+C_2}
{r_\pi C_1 C_2}
$$

$$
A =
\frac{z}{p_1p_2}
$$

$$
B =
\frac{z-p_1}
{p_1(p_1-p_2)}
$$

$$
G =
\frac{z-p_2}
{p_2(p_2-p_1)}
$$

这里用 $G$ 表示代码里的第三个 PI coefficient，避免和 Liberty delay $D$ 混淆。

## 10. Useful PI Coefficient Differentials

$$
\dot{z}_1 =
-z_1
\left(
\frac{\dot{r}_\pi}{r_\pi}
+ \frac{\dot{C}_1}{C_1}
\right)
$$

$$
\dot{k}_0 =
-k_0
\left(
\frac{\dot{r}_d}{r_d}
+ \frac{\dot{C}_2}{C_2}
\right)
$$

$$
\dot{a} =
a
\left(
\frac{\dot{r}_\pi}{r_\pi}
+ \frac{\dot{r}_d}{r_d}
+ \frac{\dot{C}_1}{C_1}
+ \frac{\dot{C}_2}{C_2}
\right)
$$

$$
\dot{b} =
\dot{r}_d(C_1+C_2)
+ r_d(\dot{C}_1+\dot{C}_2)
+ \dot{r}_\pi C_1
+ r_\pi\dot{C}_1
$$

$$
\dot{\Delta} =
2b\dot{b} - 4\dot{a}
$$

$$
\dot{g} =
\frac{\dot{\Delta}}{2g}
$$

$$
\dot{p}_1 =
\frac{\dot{b}+\dot{g}}{2a}
- p_1\frac{\dot{a}}{a}
$$

$$
\dot{p}_2 =
\frac{\dot{b}-\dot{g}}{2a}
- p_2\frac{\dot{a}}{a}
$$

其余 $\dot{k}_1,\dot{k}_2,\dot{k}_3,\dot{k}_4,\dot{A},\dot{B},\dot{G}$ 直接按乘法、除法、链式法则展开即可。

例如：

$$
\dot{k}_2 =
\frac{\dot{z}_1}{p_1p_2}
- k_2
\left(
\frac{\dot{p}_1}{p_1}
+ \frac{\dot{p}_2}{p_2}
\right)
$$

## 11. 从这里开始先看整体数据流

从这一节开始，不要先盯着公式看。先记住 DMP PI model 对一个 output net 做了两件事：

1. 把整棵 RC tree 在 driver root 处 reduce 成一个 PI load：

$$
(C_1, C_2, r_\pi)
$$

2. 对每个 sink $k$，单独保留一个 sink Elmore：

$$
E_k
$$

所以一段 segment RC 参数 $\theta$ 影响 timing 时，天然有两条链：

$$
\theta
\rightarrow
C_1,C_2,r_\pi
\rightarrow
\text{driver waveform}
$$

以及：

$$
\theta
\rightarrow
E_k
\rightarrow
\text{sink/load waveform}
$$

DMP PI 的 net sink delay 不是简单的 $E_k$，而是：

$$
\text{driver waveform}
\xrightarrow{E_k}
\text{sink waveform}
\xrightarrow{threshold crossing}
\text{delay/slew}
$$

因此全量导数必须同时包含：

- root PI 改变 driver waveform 的影响。
- sink Elmore 改变 load waveform 的影响。

## 12. 变量表

从 `## 11` 往后主要用这些变量。

| 符号 | 含义 |
|---|---|
| $\theta$ | 要求导的 segment 参数，可以是 $R_s$ 或 $C_s$ |
| $S$ | cell input slew，即 $S_{in}$ |
| $C_1,C_2,r_\pi$ | output net root 处的 PI load 参数 |
| $E_k$ | root 到 sink $k$ 的 Elmore delay |
| $r_d$ | DMP 估计出来的 driver resistance |
| $c$ | $c_{eff}$，PI solve 里的 effective capacitance |
| $h$ | $dt$，driver ramp width |
| $t_0$ | driver ramp start time shift |
| $D(S,c)$ | Liberty delay LUT |
| $U(S,c)$ | Liberty slew LUT |
| $k_0,k_1,k_2,k_3,k_4,p_1,p_2$ | PI driver waveform coefficients |
| $p_3$ | sink load pole，$p_3=1/E_k$ |
| $T_\alpha$ | driver waveform crossing time, $V_o(T_\alpha)=\alpha$ |
| $L_\alpha$ | sink/load waveform crossing time, $V_l(L_\alpha)=\alpha$ |
| $v_l,v_{th},v_h$ | lower / threshold / upper voltage thresholds |

导数记号：

$$
\dot{x}=\frac{dx}{d\theta}
$$

从前面 RC 推导结束后，我们认为已经有：

$$
\dot{C}_1,\quad
\dot{C}_2,\quad
\dot{r}_\pi,\quad
\dot{E}_k
$$

如果当前 segment 在本 cell 的 output net 上，通常：

$$
\dot{S}=0
$$

如果做整条 path 的链式导数，$\dot{S}$ 来自上游。

## 13. Step 1: 先求 driver resistance 导数

DMP 先用总电容：

$$
C_{tot}=C_1+C_2
$$

查 Liberty delay：

$$
d_1=D(S,C_{tot})
$$

再用一个小扰动：

$$
\Delta C=\frac{10^{-15}}{cap\_unit}
$$

$$
d_2=D(S,C_{tot}+\Delta C)
$$

估计 driver resistance：

$$
r_d=-\ln(v_{th})\frac{|d_1-d_2|}{\Delta C}
$$

固定当前符号：

$$
\sigma=\operatorname{sign}(d_1-d_2)
$$

则：

$$
\dot{r}_d=
-\ln(v_{th})\frac{\sigma}{\Delta C}
(\dot{d}_1-\dot{d}_2)
$$

其中：

$$
\dot{C}_{tot}=\dot{C}_1+\dot{C}_2
$$

$$
\dot{d}_1=
D_S(S,C_{tot})\dot{S}
+
D_C(S,C_{tot})\dot{C}_{tot}
$$

$$
\dot{d}_2=
D_S(S,C_{tot}+\Delta C)\dot{S}
+
D_C(S,C_{tot}+\Delta C)\dot{C}_{tot}
$$

这一节输出：

$$
\dot{r}_d
$$

## 14. Step 2: 为什么要解 $t_0,h,c$

DMP PI model 不是直接用 Liberty table delay 当作所有 waveform 信息。它要构造一个 driver waveform，使它既符合 Liberty table，又能驱动 PI RC load。

未知数是：

$$
x=
\begin{bmatrix}
t_0\\
h\\
c
\end{bmatrix}
=
\begin{bmatrix}
t_0\\
dt\\
c_{eff}
\end{bmatrix}
$$

这三个量的直观意义：

- $t_0$：driver waveform 的时间平移。
- $h=dt$：driver ramp 的宽度，也决定 slew 形状。
- $c=c_{eff}$：用哪个 effective capacitance 去查 Liberty delay/slew。

DMP 通过三个方程解这三个未知数。

## 15. Step 3: 三个方程分别是什么意思

先由 Liberty table 定义：

$$
t_{vth}=D(S,c)
$$

$$
u=U(S,c)
$$

$$
s_{meas}=u\cdot derate
$$

$$
t_{vl}=t_{vth}-s_{meas}\frac{v_{th}-v_l}{v_h-v_l}
$$

$$
c_{time}=\frac{s_{meas}}{v_h-v_l}
$$

然后 DMP 解：

$$
F(x,P)=0
$$

其中：

$$
P=(C_1,C_2,r_\pi,r_d,S)
$$

三个方程是：

### 15.1 方程 1: PI current 和 effective-cap current 匹配

$$
F_0=
I_{pi}(h,c_{time},C_1,C_2,r_\pi,r_d)
-
I_{ceff}(h,c_{time},c,r_d)
=0
$$

意思是：用 PI load 算出来的等效电流，要和用 $c_{eff}$ 算出来的电流一致。

这条方程主要决定：

$$
c=c_{eff}
$$

应该是多少。

### 15.2 方程 2: 在 table delay 时间达到 output threshold

$$
F_1=
Y_{cap}(t_{vth};t_0,h,r_d,c)-v_{th}=0
$$

意思是：用 $r_d$ 和 $c_{eff}$ 构造出来的 simple RC ramp，在 Liberty table delay 给出的时间 $t_{vth}$，应该正好达到 $v_{th}$。

### 15.3 方程 3: 在 lower threshold 时间达到 lower threshold

$$
F_2=
Y_{cap}(t_{vl};t_0,h,r_d,c)-v_l=0
$$

意思是：同一个 simple RC ramp，在 $t_{vl}$ 应该达到 $v_l$。

这三条方程一起确定：

$$
t_0,
\qquad
h,
\qquad
c
$$

## 16. Step 4: 不要求 Newton 过程导数，只对最终方程求导

这是关键点。代码里用 Newton 解 $F(x,P)=0$，但解析导数不需要展开 Newton 每次迭代。

因为最终解满足：

$$
F(x,P)=0
$$

对 $\theta$ 求导：

$$
\frac{\partial F}{\partial x}\dot{x}
+
\frac{\partial F}{\partial P}\dot{P}=0
$$

定义：

$$
J_x=\frac{\partial F}{\partial x}
$$

则：

$$
\dot{x}=-J_x^{-1}\frac{\partial F}{\partial P}\dot{P}
$$

也就是：

$$
\begin{bmatrix}
\dot{t}_0\\
\dot{h}\\
\dot{c}
\end{bmatrix}
=
-J_x^{-1}
\left(
F_{C_1}\dot{C}_1
+F_{C_2}\dot{C}_2
+F_{r_\pi}\dot{r}_\pi
+F_{r_d}\dot{r}_d
+F_S\dot{S}
\right)
$$

这里的下标不是某个新的物理量，而是偏导记号。

先记住 $F$ 不是一个标量，而是三个方程组成的向量：

$$
F =
\begin{bmatrix}
F_0\\
F_1\\
F_2
\end{bmatrix}
$$

所以 $F_{C_1}$ 表示整个向量 $F$ 对标量 $C_1$ 的偏导：

$$
F_{C_1}
=
\frac{\partial F}{\partial C_1}
=
\begin{bmatrix}
\frac{\partial F_0}{\partial C_1}\\
\frac{\partial F_1}{\partial C_1}\\
\frac{\partial F_2}{\partial C_1}
\end{bmatrix}
$$

同理：

$$
F_{C_2}
=
\begin{bmatrix}
\frac{\partial F_0}{\partial C_2}\\
\frac{\partial F_1}{\partial C_2}\\
\frac{\partial F_2}{\partial C_2}
\end{bmatrix},
\qquad
F_{r_\pi}
=
\begin{bmatrix}
\frac{\partial F_0}{\partial r_\pi}\\
\frac{\partial F_1}{\partial r_\pi}\\
\frac{\partial F_2}{\partial r_\pi}
\end{bmatrix}
$$

以及：

$$
F_{r_d}
=
\begin{bmatrix}
\frac{\partial F_0}{\partial r_d}\\
\frac{\partial F_1}{\partial r_d}\\
\frac{\partial F_2}{\partial r_d}
\end{bmatrix},
\qquad
F_S
=
\begin{bmatrix}
\frac{\partial F_0}{\partial S}\\
\frac{\partial F_1}{\partial S}\\
\frac{\partial F_2}{\partial S}
\end{bmatrix}
$$

因此括号里的东西是一个 $3\times1$ 向量：

$$
F_{C_1}\dot{C}_1
+F_{C_2}\dot{C}_2
+F_{r_\pi}\dot{r}_\pi
+F_{r_d}\dot{r}_d
+F_S\dot{S}
=
\begin{bmatrix}
\dot{F}_0\\
\dot{F}_1\\
\dot{F}_2
\end{bmatrix}_{\text{caused by }P}
$$

它表示：如果先把 $t_0,h,c$ 固定住，只让参数 $C_1,C_2,r_\pi,r_d,S$ 变化，那么三个方程 $F_0,F_1,F_2$ 会分别变化多少。

而 $J_x$ 是 $F$ 对未知数 $x=(t_0,h,c)$ 的 Jacobian：

$$
J_x =
\frac{\partial F}{\partial x}
=
\begin{bmatrix}
\frac{\partial F_0}{\partial t_0}
& \frac{\partial F_0}{\partial h}
& \frac{\partial F_0}{\partial c}\\
\frac{\partial F_1}{\partial t_0}
& \frac{\partial F_1}{\partial h}
& \frac{\partial F_1}{\partial c}\\
\frac{\partial F_2}{\partial t_0}
& \frac{\partial F_2}{\partial h}
& \frac{\partial F_2}{\partial c}
\end{bmatrix}
$$

所以这条式子的意思是：

1. segment RC 先让参数 $P=(C_1,C_2,r_\pi,r_d,S)$ 变动；
2. 这些参数变动会让三个方程 $F_0,F_1,F_2$ 不再等于 0；
3. 为了让 $F(x,P)=0$ 继续成立，解出来的 $t_0,h,c$ 必须跟着变；
4. $-J_x^{-1}$ 就是在算“$t_0,h,c$ 需要怎么补偿这些参数变化”。


这里：

$$
\dot{P}=
(\dot{C}_1,\dot{C}_2,\dot{r}_\pi,\dot{r}_d,\dot{S})
$$

实际实现时，$J_x$ 就是当前 PI solve 方程对 $t_0,h,c$ 的 Jacobian。概念上它是一个 $3\times3$ 矩阵。

这一节输出：

$$
\dot{t}_0,
\qquad
\dot{h},
\qquad
\dot{c}
$$

## 17. Step 5: PI coefficients 及其导数

解出 $t_0,h,c$ 后，还需要把 PI load 转成 waveform coefficients。

PI coefficients 由：

$$
C_1,C_2,r_\pi,r_d
$$

决定。

主要公式是：

$$
z_1=\frac{1}{r_\pi C_1}
$$

$$
k_0=\frac{1}{r_d C_2}
$$

$$
a=r_\pi r_d C_1C_2
$$

$$
b=r_d(C_1+C_2)+r_\pi C_1
$$

$$
\Delta=b^2-4a
$$

$$
g=\sqrt{\Delta}
$$

$$
p_1=\frac{b+g}{2a}
$$

$$
p_2=\frac{b-g}{2a}
$$

$$
k_2=\frac{z_1}{p_1p_2}
$$

$$
k_1=\frac{1-k_2(p_1+p_2)}{p_1p_2}
$$

$$
k_4=\frac{k_1p_1+k_2}{p_2-p_1}
$$

$$
k_3=-k_1-k_4
$$

这些式子都只是普通代数式，所以导数直接链式展开即可。例如：

$$
\dot{a}=a\left(
\frac{\dot{r}_\pi}{r_\pi}
+\frac{\dot{r}_d}{r_d}
+\frac{\dot{C}_1}{C_1}
+\frac{\dot{C}_2}{C_2}
\right)
$$

$$
\dot{b}=\dot{r}_d(C_1+C_2)
+r_d(\dot{C}_1+\dot{C}_2)
+\dot{r}_\pi C_1
+r_\pi\dot{C}_1
$$

$$
\dot{\Delta}=2b\dot{b}-4\dot{a}
$$

$$
\dot{g}=\frac{\dot{\Delta}}{2g}
$$

$$
\dot{p}_1=
\frac{\dot{b}+\dot{g}}{2a}
-p_1\frac{\dot{a}}{a}
$$

$$
\dot{p}_2=
\frac{\dot{b}-\dot{g}}{2a}
-p_2\frac{\dot{a}}{a}
$$

这一节输出：

$$
\dot{k}_0,\dot{k}_1,\dot{k}_2,\dot{k}_3,\dot{k}_4,
\dot{p}_1,\dot{p}_2
$$

## 18. Step 6: driver waveform 长什么样

定义 driver base waveform：

$$
B_o(t)=
k_0
\left[
 k_1+k_2t+k_3e^{-p_1t}+k_4e^{-p_2t}
\right]
$$

它的时间导数是：

$$
B_o'(t)=
k_0
\left[
 k_2-k_3p_1e^{-p_1t}-k_4p_2e^{-p_2t}
\right]
$$

真正的 driver output waveform 是 finite ramp 形式。令：

$$
\tau=T-t_0
$$

如果 $0<\tau\le h$：

$$
V_o(T)=\frac{B_o(\tau)}{h}
$$

如果 $\tau>h$：

$$
V_o(T)=\frac{B_o(\tau)-B_o(\tau-h)}{h}
$$

## 19. Step 7: driver crossing 时间怎么求导

driver crossing $T_\alpha$ 定义为：

$$
V_o(T_\alpha)=\alpha
$$

其中：

$$
\alpha\in\{v_l,v_{th},v_h\}
$$

对 $\theta$ 求导：

$$
V_{o,T}\dot{T}_\alpha+
\dot{V}_{o,explicit}=0
$$

所以：

$$
\dot{T}_\alpha=
-\frac{\dot{V}_{o,explicit}}{V_{o,T}}
$$

这里的意思是：

- $V_{o,T}$ 是 waveform 对时间的斜率。
- $\dot{V}_{o,explicit}$ 是固定 crossing time 不动时，waveform 因参数变化产生的变化。

### 19.1 为什么会有 $\dot{V}_{o,explicit}$ 这一项

这里没有违反链式法则。恰恰相反，$\dot{V}_{o,explicit}$ 就是链式法则里“除了 $T_\alpha$ 之外，其他参数变化”的那一项。

把 driver waveform 写成一个多变量函数：

$$
V_o = V_o(T, q)
$$

其中：

$$
q=(t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2)
$$

crossing equation 是：

$$
V_o(T_\alpha(\theta), q(\theta))=\alpha
$$

注意这里有两类东西依赖 $\theta$：

1. crossing time 本身：

$$
T_\alpha=T_\alpha(\theta)
$$

2. waveform 参数：

$$
q=q(\theta)
$$

所以对 $\theta$ 求全导数：

$$
\frac{d}{d\theta}V_o(T_\alpha(\theta),q(\theta))
=
\frac{\partial V_o}{\partial T}\dot{T}_\alpha
+
\frac{\partial V_o}{\partial q}\dot{q}
$$

因为右边 threshold $\alpha$ 是常数：

$$
\frac{d\alpha}{d\theta}=0
$$

所以：

$$
V_{o,T}\dot{T}_\alpha
+
V_{o,q}\dot{q}=0
$$

这里定义：

$$
\dot{V}_{o,explicit}=V_{o,q}\dot{q}
$$

因此：

$$
V_{o,T}\dot{T}_\alpha+
\dot{V}_{o,explicit}=0
$$

也就是说，$\dot{V}_{o,explicit}$ 不是额外加出来的项，它就是链式法则中的：

$$
\frac{\partial V_o}{\partial q}\dot{q}
$$

只是为了和 crossing time 的变化区分开，单独起了名字。

### 19.2 一个最简单的例子

假设有一个简单 crossing 方程：

$$
g(T,a)=T-a=0
$$

显然解是：

$$
T=a
$$

如果 $a=a(\theta)$，那么正确导数是：

$$
\dot{T}=\dot{a}
$$

按上面的写法：

$$
g_T\dot{T}+g_a\dot{a}=0
$$

其中：

$$
g_T=1,
\qquad
 g_a=-1
$$

所以：

$$
\dot{T}-\dot{a}=0
$$

$$
\dot{T}=\dot{a}
$$

这里的：

$$
g_a\dot{a}
$$

就对应 DMP 里的 $\dot{V}_{o,explicit}$。如果漏掉它，只写：

$$
g_T\dot{T}=0
$$

就会得到错误结论：

$$
\dot{T}=0
$$

### 19.3 回到 DMP driver waveform

DMP 里 driver crossing equation 是：

$$
V_o(T_\alpha,t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2)=\alpha
$$

所以全导数完整写开是：

$$
V_{o,T}\dot{T}_\alpha
+V_{o,t_0}\dot{t}_0
+V_{o,h}\dot{h}
+V_{o,k_0}\dot{k}_0
+V_{o,k_1}\dot{k}_1
+V_{o,k_2}\dot{k}_2
+V_{o,k_3}\dot{k}_3
+V_{o,k_4}\dot{k}_4
+V_{o,p_1}\dot{p}_1
+V_{o,p_2}\dot{p}_2
=0
$$

为了不每次都写这么长，定义：

$$
\dot{V}_{o,explicit}
=
V_{o,t_0}\dot{t}_0
+V_{o,h}\dot{h}
+V_{o,k_0}\dot{k}_0
+V_{o,k_1}\dot{k}_1
+V_{o,k_2}\dot{k}_2
+V_{o,k_3}\dot{k}_3
+V_{o,k_4}\dot{k}_4
+V_{o,p_1}\dot{p}_1
+V_{o,p_2}\dot{p}_2
$$

于是就得到短写法：

$$
V_{o,T}\dot{T}_\alpha+
\dot{V}_{o,explicit}=0
$$

以及：

$$
\dot{T}_\alpha=-\frac{\dot{V}_{o,explicit}}{V_{o,T}}
$$

所以 $\dot{V}_{o,explicit}$ 的来源就是：segment RC 改变了 $C_1,C_2,r_\pi$，进一步改变了 $r_d,t_0,h,c,k_i,p_i$，导致在同一个时间 $T_\alpha$ 上，整条 driver waveform 本身上下移动或变形。

### 19.4 Crossing 在 ramp 内

如果：

$$
0<\tau\le h
$$

则：

$$
V_{o,T}=\frac{B_o'(\tau)}{h}
$$

固定 $T_\alpha$ 时：

$$
\dot{\tau}=-\dot{t}_0
$$

于是：

$$
\dot{V}_{o,explicit}=
\frac{
\dot{B}_{o,c}(\tau)-B_o'(\tau)\dot{t}_0
}{h}
-
\frac{B_o(\tau)}{h^2}\dot{h}
$$

其中 $\dot{B}_{o,c}$ 表示只对 $k_i,p_i$ 求导，不对时间 $t$ 求导：

$$
\dot{B}_{o,c}(t)=
\dot{k}_0
\left[k_1+k_2t+k_3e^{-p_1t}+k_4e^{-p_2t}\right]
+
k_0
\left[
\dot{k}_1+t\dot{k}_2+e^{-p_1t}\dot{k}_3+e^{-p_2t}\dot{k}_4
-tk_3e^{-p_1t}\dot{p}_1
-tk_4e^{-p_2t}\dot{p}_2
\right]
$$

### 19.5 Crossing 在 ramp 后

如果：

$$
\tau>h
$$

则：

$$
V_{o,T}=\frac{B_o'(\tau)-B_o'(\tau-h)}{h}
$$

固定 $T_\alpha$ 时：

$$
\dot{\tau}=-\dot{t}_0
$$

$$
\dot{(\tau-h)}=-\dot{t}_0-\dot{h}
$$

于是：

$$
\dot{V}_{o,explicit}=
\frac{
\dot{B}_{o,c}(\tau)-B_o'(\tau)\dot{t}_0
-
\dot{B}_{o,c}(\tau-h)
+B_o'(\tau-h)(\dot{t}_0+\dot{h})
}{h}
-
\frac{B_o(\tau)-B_o(\tau-h)}{h^2}\dot{h}
$$

再代入：

$$
\dot{T}_\alpha=-\frac{\dot{V}_{o,explicit}}{V_{o,T}}
$$

这一节输出：

$$
\dot{T}_{v_l},\quad
\dot{T}_{v_{th}},\quad
\dot{T}_{v_h}
$$

## 20. Step 8: cell delay 和 cell slew

PI branch 的 cell delay 是：

$$
d_{cell}=D(S,c)
$$

所以：

$$
\dot{d}_{cell}=D_S\dot{S}+D_C\dot{c}
$$

cell output slew 来自 driver waveform crossing：

$$
s_{cell}=\frac{T_{v_h}-T_{v_l}}{derate}
$$

所以：

$$
\dot{s}_{cell}=\frac{\dot{T}_{v_h}-\dot{T}_{v_l}}{derate}
$$

## 21. Step 9: load waveform 为什么同时依赖 driver 和 Elmore

对 sink $k$，load waveform 由两个东西决定：

1. driver waveform，也就是：

$$
t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2
$$

2. sink Elmore：

$$
E_k
$$

令：

$$
p_3=\frac{1}{E_k}
$$

则：

$$
\dot{p}_3=-\frac{\dot{E}_k}{E_k^2}
$$

定义 load base waveform：

$$
B_l(t)=
\ell_1+t+
\ell_3e^{-p_1t}+
\ell_4e^{-p_2t}+
\ell_5e^{-p_3t}
$$

其中：

$$
\ell_1=k_0\left(k_1-\frac{k_2}{p_3}\right)
$$

$$
\ell_3=-\frac{p_3k_0k_3}{p_1-p_3}
$$

$$
\ell_4=-\frac{p_3k_0k_4}{p_2-p_3}
$$

$$
\ell_5=k_0\left(
\frac{k_2}{p_3}-k_1
+\frac{p_3k_3}{p_1-p_3}
+\frac{p_3k_4}{p_2-p_3}
\right)
$$

这里可以清楚看到：

- $\ell_i$ 依赖 $k_i,p_1,p_2$，所以依赖 driver waveform。
- $\ell_i$ 也依赖 $p_3=1/E_k$，所以依赖 sink Elmore。

因此：

$$
\dot{B}_{l,c}(t)
=
\dot{B}_{l,c}^{driver}(t)
+
\dot{B}_{l,c}^{elmore}(t)
$$

这句话不是说有东西做不出来，而是说完整导数必须把两部分都加上。

## 22. Step 10: load crossing 时间怎么求导

真正的 sink waveform 也是 finite ramp 形式。令：

$$
\tau=L-t_0
$$

如果 $0<\tau\le h$：

$$
V_l(L)=\frac{B_l(\tau)}{h}
$$

如果 $\tau>h$：

$$
V_l(L)=\frac{B_l(\tau)-B_l(\tau-h)}{h}
$$

load crossing $L_\alpha$ 定义为：

$$
V_l(L_\alpha)=\alpha
$$

对 $\theta$ 求导：

$$
V_{l,L}\dot{L}_\alpha+
\dot{V}_{l,explicit}=0
$$

所以：

$$
\dot{L}_\alpha=-\frac{\dot{V}_{l,explicit}}{V_{l,L}}
$$

也可以写成更直观的全量分解：

$$
\dot{L}_\alpha=
-\frac{V_{l,w}^{(\alpha)}\dot{w}+V_{l,E}^{(\alpha)}\dot{E}_k}{V_{l,L}^{(\alpha)}}
$$

其中：

$$
w=(t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2)
$$

这就是完整的 load crossing 导数。

### 22.1 Crossing 在 ramp 内

如果：

$$
0<\tau\le h
$$

则：

$$
V_{l,L}=\frac{B_l'(\tau)}{h}
$$

$$
\dot{V}_{l,explicit}=
\frac{
\dot{B}_{l,c}(\tau)-B_l'(\tau)\dot{t}_0
}{h}
-
\frac{B_l(\tau)}{h^2}\dot{h}
$$

### 22.2 Crossing 在 ramp 后

如果：

$$
\tau>h
$$

则：

$$
V_{l,L}=\frac{B_l'(\tau)-B_l'(\tau-h)}{h}
$$

$$
\dot{V}_{l,explicit}=
\frac{
\dot{B}_{l,c}(\tau)-B_l'(\tau)\dot{t}_0
-
\dot{B}_{l,c}(\tau-h)
+B_l'(\tau-h)(\dot{t}_0+\dot{h})
}{h}
-
\frac{B_l(\tau)-B_l(\tau-h)}{h^2}\dot{h}
$$

再代入：

$$
\dot{L}_\alpha=-\frac{\dot{V}_{l,explicit}}{V_{l,L}}
$$

这一节输出：

$$
\dot{L}_{v_l},\quad
\dot{L}_{v_{th}},\quad
\dot{L}_{v_h}
$$

## 23. Step 11: net sink delay 的完整导数

raw net sink delay 是：

$$
d_{net,k}^{raw}=L_{v_{th}}-T_{v_{th}}
$$

所以：

$$
\dot{d}_{net,k}^{raw}=\dot{L}_{v_{th}}-\dot{T}_{v_{th}}
$$

而：

$$
\dot{L}_{v_{th}}=
-\frac{V_{l,w}\dot{w}+V_{l,E}\dot{E}_k}{V_{l,L}}
$$

$$
\dot{T}_{v_{th}}=
-\frac{V_{o,w}\dot{w}}{V_{o,T}}
$$

代进去：

$$
\dot{d}_{net,k}^{raw}=
-\frac{V_{l,w}\dot{w}+V_{l,E}\dot{E}_k}{V_{l,L}}
+
\frac{V_{o,w}\dot{w}}{V_{o,T}}
$$

展开成三项：

$$
\dot{d}_{net,k}^{raw}=
-\frac{V_{l,E}\dot{E}_k}{V_{l,L}}
-\frac{V_{l,w}\dot{w}}{V_{l,L}}
+\frac{V_{o,w}\dot{w}}{V_{o,T}}
$$

这三项的含义是：

1. $-V_{l,E}\dot{E}_k/V_{l,L}$：sink Elmore 变化对 load crossing 的影响。
2. $-V_{l,w}\dot{w}/V_{l,L}$：driver waveform 变化对 load crossing 的影响。
3. $+V_{o,w}\dot{w}/V_{o,T}$：driver threshold crossing $T_{v_{th}}$ 本身变化带来的抵消项。

这就是 PI model 下 net sink delay 对 segment RC 的完整局部解析导数。

## 24. Step 12: net sink slew 的完整导数

raw net sink slew 是：

$$
s_{net,k}^{raw}=
\frac{L_{v_h}-L_{v_l}}{driver\_derate}
$$

所以：

$$
\dot{s}_{net,k}^{raw}=
\frac{\dot{L}_{v_h}-\dot{L}_{v_l}}{driver\_derate}
$$

其中每个 crossing 都用同一个公式：

$$
\dot{L}_\alpha=
-\frac{V_{l,w}^{(\alpha)}\dot{w}+V_{l,E}^{(\alpha)}\dot{E}_k}{V_{l,L}^{(\alpha)}}
$$

所以 slew 也同时包含 driver waveform term 和 sink Elmore term。

## 25. Step 13: threshold adjust

如果 driver/load threshold 不同，DMP 会做 threshold adjust。

定义：

$$
\beta=
\frac{load\_v_{th}-driver\_v_{th}}{driver\_v_h-driver\_v_l}
$$

$$
\gamma=
\frac{(load\_v_h-load\_v_l)/load\_derate}
{(driver\_v_h-driver\_v_l)/driver\_derate}
$$

rise 时：

$$
sign=+1
$$

fall 时：

$$
sign=-1
$$

最终：

$$
d_{net,k}=d_{net,k}^{raw}+sign\cdot\beta\cdot s_{net,k}^{raw}
$$

$$
s_{net,k}=\gamma s_{net,k}^{raw}
$$

导数：

$$
\dot{d}_{net,k}=\dot{d}_{net,k}^{raw}+sign\cdot\beta\cdot\dot{s}_{net,k}^{raw}
$$

$$
\dot{s}_{net,k}=\gamma\dot{s}_{net,k}^{raw}
$$

如果 driver/load threshold 相同，则：

$$
\beta=0,
\qquad
\gamma=1
$$

## 26. 按实现顺序的完整计算清单

对每个 segment 参数 $\theta$，PI branch 的全量解析导数可以按下面顺序算。

### 26.1 RC derivative

自底向上算：

$$
\dot{M}_v,
\quad
\dot{N}_v,
\quad
\dot{P}_v
$$

对每个 sink $k$ 算：

$$
\dot{E}_k
$$

### 26.2 Root PI derivative

由 root moments 算：

$$
\dot{C}_1,
\quad
\dot{C}_2,
\quad
\dot{r}_\pi
$$

### 26.3 Driver model derivative

先算：

$$
\dot{r}_d
$$

再通过 implicit differentiation 解：

$$
\dot{t}_0,
\quad
\dot{h},
\quad
\dot{c}
$$

### 26.4 Driver waveform derivative

算：

$$
\dot{k}_0,
\dot{k}_1,
\dot{k}_2,
\dot{k}_3,
\dot{k}_4,
\dot{p}_1,
\dot{p}_2
$$

再算：

$$
\dot{T}_{v_l},
\quad
\dot{T}_{v_{th}},
\quad
\dot{T}_{v_h}
$$

### 26.5 Cell output

$$
\dot{d}_{cell}=D_S\dot{S}+D_C\dot{c}
$$

$$
\dot{s}_{cell}=\frac{\dot{T}_{v_h}-\dot{T}_{v_l}}{derate}
$$

### 26.6 Sink load waveform derivative

对每个 sink $k$，用 $\dot{E}_k$ 和 driver waveform derivative 算：

$$
\dot{L}_{v_l},
\quad
\dot{L}_{v_{th}},
\quad
\dot{L}_{v_h}
$$

### 26.7 Net sink output

$$
\dot{d}_{net,k}^{raw}=\dot{L}_{v_{th}}-\dot{T}_{v_{th}}
$$

$$
\dot{s}_{net,k}^{raw}=\frac{\dot{L}_{v_h}-\dot{L}_{v_l}}{driver\_derate}
$$

最后加 threshold adjust：

$$
\dot{d}_{net,k}=\dot{d}_{net,k}^{raw}+sign\cdot\beta\cdot\dot{s}_{net,k}^{raw}
$$

$$
\dot{s}_{net,k}=\gamma\dot{s}_{net,k}^{raw}
$$

## 27. 最关键的一句话

PI DMP 的 net sink delay 不是：

$$
d_{net,k}=E_k
$$

而是：

$$
d_{net,k}=L_{v_{th}}-T_{v_{th}}
$$

其中 $L_{v_{th}}$ 由 driver waveform 和 sink Elmore 共同决定。

因此对 segment RC 的导数一定是：

$$
\text{segment RC}
\rightarrow
\text{root PI / driver waveform}
\rightarrow
T_\alpha,L_\alpha
$$

加上：

$$
\text{segment RC}
\rightarrow
E_k
\rightarrow
L_\alpha
$$

两条链的和。

## 28. 如果 $J_x$ 不可逆怎么办

前面用了：

$$
\dot{x}=-J_x^{-1}\dot{F}_P
$$

这里隐含了一个条件：

$$
J_x=\frac{\partial F}{\partial(t_0,h,c)}
$$

必须可逆，或者至少数值上足够 well-conditioned。

### 28.1 数学含义

如果 $J_x$ 不可逆，说明在当前 operating point 附近，三个方程：

$$
F_0=0,
\qquad
F_1=0,
\qquad
F_2=0
$$

不能唯一地决定：

$$
t_0,
\qquad
h,
\qquad
c
$$

也就是说，局部上可能存在以下情况：

- 多组 $t_0,h,c$ 都满足同一组方程；
- 或者某个方向上 perturbation 后没有稳定的一阶响应；
- 或者 crossing / waveform slope 太平，导致小参数变化引起很大的 $t_0,h,c$ 变化；
- 或者当前点正好在 PI model 的退化位置。

这时严格的 ordinary derivative：

$$
\frac{d(t_0,h,c)}{d\theta}
$$

不存在或者不唯一。

所以理论上，$J_x$ singular 时不能继续说“PI branch 的解析导数就是某个唯一值”。

### 28.2 常见退化原因

在 DMP PI model 里，$J_x$ 可能变得 singular 或 ill-conditioned 的原因包括：

1. waveform crossing slope 太小：

$$
V_{o,T}\approx 0
$$

或：

$$
Y_{cap,T}\approx 0
$$

这会让 threshold crossing 对参数变化极端敏感。

2. PI poles 退化：

$$
p_1\approx p_2
$$

此时很多 coefficient 里有：

$$
\frac{1}{p_2-p_1}
$$

会变得非常大。

3. load pole 和 driver pole 接近：

$$
p_3\approx p_1
\quad\text{or}\quad
p_3\approx p_2
$$

因为 load waveform coefficient 里有：

$$
\frac{1}{p_1-p_3},
\qquad
\frac{1}{p_2-p_3}
$$

4. PI load 退化，例如：

$$
C_2\approx 0
$$

或：

$$
r_\pi\le 0
$$

5. Liberty LUT 局部太平，导致 $D(S,c),U(S,c)$ 对 $c$ 或 $S$ 的敏感度不足，三个方程局部相关。

### 28.3 工程处理方式

如果只讨论纯数学 PI branch：

$$
J_x \text{ singular}
\quad\Rightarrow\quad
\text{PI analytic derivative is undefined or non-unique.}
$$

但工程实现通常不能直接崩掉。可以按下面优先级处理。

#### 28.3.1 判断 condition number

不要只判断 det 是否等于 0。应该判断 condition number 或求解残差。

例如：

$$
\kappa(J_x)=\|J_x\|\|J_x^{-1}\|
$$

如果：

$$
\kappa(J_x) > \kappa_{max}
$$

就认为这个点的解析导数不可靠。

实际实现也可以不用显式算 $J_x^{-1}$，而是解线性方程：

$$
J_x\dot{x}=-\dot{F}_P
$$

然后检查 solve residual：

$$
\|J_x\dot{x}+\dot{F}_P\|
$$

#### 28.3.2 Damped solve

如果 $J_x$ 病态但还想给出一个稳定方向，可以用 Levenberg-Marquardt / Tikhonov damping：

$$
(J_x^TJ_x+\lambda I)\dot{x}
=
-J_x^T\dot{F}_P
$$

于是：

$$
\dot{x}
=-(J_x^TJ_x+\lambda I)^{-1}J_x^T\dot{F}_P
$$

这里 $\lambda>0$ 控制保守程度。

这不是严格 PI branch ordinary derivative，而是一个 regularized sensitivity。

#### 28.3.3 Pseudo-inverse

也可以用 Moore-Penrose pseudo-inverse：

$$
\dot{x}=-J_x^{+}\dot{F}_P
$$

这会给出最小范数解。

同样，这在 singular 点不是唯一真实导数，而是选择了一个 convention。

#### 28.3.4 回退到 finite difference

对 debug 或验证，最直接的兜底是有限差分：

$$
\frac{y(\theta+\epsilon)-y(\theta)}{\epsilon}
$$

其中 $y$ 可以是：

$$
d_{cell},\quad s_{cell},\quad d_{net,k},\quad s_{net,k}
$$

有限差分也会受分支切换影响，但它能告诉你当前整体代码路径的实际数值响应。

#### 28.3.5 回退到非 PI branch

真实 DMP 代码里 PI solve 失败或数值不合法时，会 fallback 到更简单的模型。对应导数也应该跟着 fallback：

- PI solve 正常：用 PI analytic derivative。
- PI solve singular / ill-conditioned：不要硬算 PI derivative。
- fallback 到 CAP / table model：用 fallback branch 的导数。

如果你坚持“只走 PI model”，那 singular 点只能标记为 invalid sensitivity。

### 28.4 推荐策略

对当前推导，建议采用这个定义：

1. 正常情况：

$$
\dot{x}=-J_x^{-1}\dot{F}_P
$$

2. 如果 $J_x$ ill-conditioned：

$$
\text{PI analytic derivative invalid}
$$

3. 工程上需要连续数值时，用 damped solve：

$$
\dot{x}=-(J_x^TJ_x+\lambda I)^{-1}J_x^T\dot{F}_P
$$

4. debug 时用 finite difference 对比。

最重要的是：不要把病态点算出来的巨大导数当成可靠 timing sensitivity。它通常代表当前 PI parameterization 已经退化，或者这个点附近 DMP 分支本身不光滑。

## 29. 把 $F_0,F_1,F_2$ 具体展开

前面为了讲清楚结构，把 driver solve 写成：

$$
F(x,P)=0
$$

这里把 $F_0,F_1,F_2$ 的核心公式展开。这样你能看到 $F_{C_1},F_{r_d},F_S$ 这些偏导到底从哪里来。

### 29.1 变量

未知数：

$$
x=(t_0,h,c)
$$

参数：

$$
P=(C_1,C_2,r_\pi,r_d,S)
$$

Liberty table：

$$
t_{vth}=D(S,c)
$$

$$
u=U(S,c)
$$

$$
s_{meas}=u\cdot derate
$$

$$
t_{vl}=t_{vth}-s_{meas}\frac{v_{th}-v_l}{v_h-v_l}
$$

$$
c_{time}=\frac{s_{meas}}{v_h-v_l}
$$

### 29.2 Simple RC ramp function $Y_{cap}$

令：

$$
\tau_{rc}=r_dc
$$

定义基础响应：

$$
y_0(x;\tau_{rc})=x-\tau_{rc}\left(1-e^{-x/\tau_{rc}}\right)
$$

其中：

$$
x=t-t_0
$$

则 simple RC ramp response 是：

如果 $x\le0$：

$$
Y_{cap}(t;t_0,h,r_d,c)=0
$$

如果 $0<x\le h$：

$$
Y_{cap}(t;t_0,h,r_d,c)=\frac{y_0(x;\tau_{rc})}{h}
$$

如果 $x>h$：

$$
Y_{cap}(t;t_0,h,r_d,c)=
\frac{y_0(x;\tau_{rc})-y_0(x-h;\tau_{rc})}{h}
$$

这个就是 DMP solve 里用来匹配 Liberty delay/slew 的 simple RC ramp。

### 29.3 方程 $F_1,F_2$

$$
F_1=Y_{cap}(t_{vth};t_0,h,r_d,c)-v_{th}
$$

$$
F_2=Y_{cap}(t_{vl};t_0,h,r_d,c)-v_l
$$

注意 $t_{vth}$ 和 $t_{vl}$ 本身依赖 $S,c$。所以例如：

$$
\frac{\partial F_1}{\partial S}
=
Y_{cap,t}\frac{\partial t_{vth}}{\partial S}
$$

其中：

$$
\frac{\partial t_{vth}}{\partial S}=D_S(S,c)
$$

而：

$$
\frac{\partial F_1}{\partial c}
$$

有两类贡献：

1. $Y_{cap}$ 自己对 $c$ 的显式依赖；
2. $t_{vth}=D(S,c)$ 也对 $c$ 变化。

所以：

$$
\frac{\partial F_1}{\partial c}
=
Y_{cap,c}
+
Y_{cap,t}D_c(S,c)
$$

这里 $Y_{cap,c}$ 表示固定 $t$ 时，simple RC response 对 $c$ 的偏导。

同理，$F_2$ 里因为 $t_{vl}$ 依赖 delay LUT 和 slew LUT：

$$
t_{vl}=D(S,c)-U(S,c)\cdot derate\cdot\frac{v_{th}-v_l}{v_h-v_l}
$$

所以：

$$
\frac{\partial t_{vl}}{\partial S}
=
D_S
-
derate\cdot\frac{v_{th}-v_l}{v_h-v_l}U_S
$$

$$
\frac{\partial t_{vl}}{\partial c}
=
D_c
-
derate\cdot\frac{v_{th}-v_l}{v_h-v_l}U_c
$$

### 29.4 方程 $F_0$

$F_0$ 是 PI current 和 effective-cap current 匹配：

$$
F_0=I_{pi}-I_{ceff}
$$

令：

$$
t_c=c_{time}
$$

PI current term：

$$
I_{pi}=
\frac{
A t_c
+
\frac{B}{p_1}\left(1-e^{-p_1t_c}\right)
+
\frac{G}{p_2}\left(1-e^{-p_2t_c}\right)
}
{r_d t_c h}
$$

Effective-cap current term：

$$
I_{ceff}=
\frac{
(r_dc)t_c
-
(r_dc)^2\left(1-e^{-t_c/(r_dc)}\right)
}
{r_d t_c h}
$$

这里的 $A,B,G$ 不是额外未知量，而是 PI pole/zero 展开后的系数。先定义：

$$
z=\frac{C_1+C_2}{r_\pi C_1C_2}
$$

$$
A=\frac{z}{p_1p_2}
$$

$$
B=\frac{z-p_1}{p_1(p_1-p_2)}
$$

$$
G=\frac{z-p_2}{p_2(p_2-p_1)}
$$

而 $p_1,p_2$ 已经由 $C_1,C_2,r_\pi,r_d$ 决定：

$$
p_1=\frac{b+\sqrt{b^2-4a}}{2a},
\qquad
p_2=\frac{b-\sqrt{b^2-4a}}{2a}
$$

$$
a=r_\pi r_d C_1C_2,
\qquad
b=r_d(C_1+C_2)+r_\pi C_1
$$

所以 $F_0$ 的偏导来源是：

- $C_1,C_2,r_\pi,r_d$ 改变 $A,B,G,p_1,p_2$；
- $r_d$ 还直接出现在分母和 $I_{ceff}$；
- $c$ 通过 $I_{ceff}$ 和 $t_c=c_{time}$ 进入；
- $S$ 通过 $t_c=c_{time}$ 进入；
- $h$ 直接出现在分母。

因此 $F_{C_1},F_{C_2},F_{r_\pi},F_{r_d},F_S$ 不是神秘符号，它们就是把上面这些显式公式分别对对应变量求偏导。

## 30. 最终输出的解析式：forward sensitivity 形式

如果选定一个 segment 参数：

$$
\theta=R_s\quad\text{or}\quad\theta=C_s
$$

那么 forward sensitivity 的最终计算链就是：

$$
\theta
\rightarrow
\dot{M},\dot{N},\dot{P},\dot{E}_k
\rightarrow
\dot{C}_1,\dot{C}_2,\dot{r}_\pi
\rightarrow
\dot{r}_d
\rightarrow
\dot{t}_0,\dot{h},\dot{c}
\rightarrow
\dot{k}_i,\dot{p}_i
\rightarrow
\dot{T}_\alpha,\dot{L}_\alpha
\rightarrow
\dot{d},\dot{s}
$$

最后输出是：

### 30.1 Cell delay

$$
\dot{d}_{cell}=D_S\dot{S}+D_c\dot{c}
$$

如果 segment 在当前 output net 上，通常 $\dot{S}=0$，所以：

$$
\dot{d}_{cell}=D_c\dot{c}
$$

### 30.2 Cell output slew

$$
\dot{s}_{cell}=\frac{\dot{T}_{v_h}-\dot{T}_{v_l}}{derate}
$$

### 30.3 Raw net sink delay

$$
\dot{d}_{net,k}^{raw}=\dot{L}_{v_{th}}-\dot{T}_{v_{th}}
$$

等价地写成 driver waveform term 和 sink Elmore term：

$$
\dot{d}_{net,k}^{raw}=
-
\frac{V_{l,E}\dot{E}_k}{V_{l,L}}
-
\frac{V_{l,w}\dot{w}}{V_{l,L}}
+
\frac{V_{o,w}\dot{w}}{V_{o,T}}
$$

其中：

$$
w=(t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2)
$$

第一项是 sink Elmore，后两项是 driver waveform。

### 30.4 Raw net sink slew

$$
\dot{s}_{net,k}^{raw}=
\frac{\dot{L}_{v_h}-\dot{L}_{v_l}}{driver\_derate}
$$

其中每个 $\dot{L}_\alpha$ 都是：

$$
\dot{L}_\alpha=
-
\frac{V_{l,w}^{(\alpha)}\dot{w}+V_{l,E}^{(\alpha)}\dot{E}_k}
{V_{l,L}^{(\alpha)}}
$$

### 30.5 Threshold adjusted net delay and slew

$$
\dot{d}_{net,k}=
\dot{d}_{net,k}^{raw}
+
sign\cdot\beta\cdot\dot{s}_{net,k}^{raw}
$$

$$
\dot{s}_{net,k}=
\gamma\dot{s}_{net,k}^{raw}
$$

## 31. forward sensitivity 和 reverse adjoint 到底哪个方向对

你说“求导数不应该是从 cell delay 回推到 RC segment 吗？”这个直觉是对的：如果你要对很多 segment 求一个 scalar output 的梯度，reverse adjoint 更合适。

但前面文档用的是 forward sensitivity。两者都是链式法则，只是计算方向不同。

### 31.1 Forward sensitivity 是什么

Forward sensitivity 是先选一个参数：

$$
\theta=R_s\quad\text{or}\quad C_s
$$

然后一路算：

$$
\dot{x}=\frac{dx}{d\theta}
$$

所以它从 segment RC 往 timing output 推：

$$
\theta\rightarrow d_{cell},s_{cell},d_{net,k},s_{net,k}
$$

优点：概念直接，适合验证某一个 segment 的导数。

缺点：如果有 $N$ 个 segment，要全量梯度就要跑 $N$ 组 sensitivity，太贵。

所以 forward sensitivity 可以在 timing forward 的同时算，但前提是你只关心少数几个 $\theta$，并且要额外携带这些 $\dot{x}$ 数组。普通 timing 本身不会自动给你全量导数。

### 31.2 Reverse adjoint 是什么

Reverse adjoint 是先正常做一次 forward timing，存下所有中间量：

$$
M,N,P,E_k,C_1,C_2,r_\pi,r_d,t_0,h,c,k_i,p_i,T_\alpha,L_\alpha
$$

然后从目标输出反向回推。

例如目标是某个 sink delay：

$$
Y=d_{net,k}
$$

先设：

$$
\bar{Y}=1
$$

然后反向传播 adjoint：

$$
\bar{x}=\frac{\partial Y}{\partial x}
$$

最后得到：

$$
\bar{R}_s=\frac{\partial Y}{\partial R_s},
\qquad
\bar{C}_s=\frac{\partial Y}{\partial C_s}
$$

这就是你说的“从 cell delay / net delay 回推到 RC segment”。

### 31.3 crossing 的 reverse 公式

如果 driver crossing 由隐式方程定义：

$$
V_o(T_\alpha,w)=\alpha
$$

forward sensitivity 是：

$$
\dot{T}_\alpha=-\frac{V_{o,w}\dot{w}}{V_{o,T}}
$$

其中：

$$
V_{o,w}=
\begin{bmatrix}
\frac{\partial V_o}{\partial w_1} &
\frac{\partial V_o}{\partial w_2} &
\cdots
\end{bmatrix}
$$

如果 reverse 里已经有 crossing time 的 adjoint $\bar{T}_\alpha$，那么对 $w$ 的贡献是：

$$
\bar{w}
\leftarrow
\bar{w}
-
\bar{T}_\alpha\frac{V_{o,w}}{V_{o,T}}
$$

load crossing 由：

$$
V_l(L_\alpha,w,E_k)=\alpha
$$

定义。forward sensitivity 是：

$$
\dot{L}_\alpha=
-\frac{V_{l,w}\dot{w}+V_{l,E}\dot{E}_k}{V_{l,L}}
$$

reverse 里对应：

$$
\bar{w}
\leftarrow
\bar{w}
-
\bar{L}_\alpha\frac{V_{l,w}}{V_{l,L}}
$$

$$
\bar{E}_k
\leftarrow
\bar{E}_k
-
\bar{L}_\alpha\frac{V_{l,E}}{V_{l,L}}
$$

这里的 $V_{o,T}$ 和 $V_{l,L}$ 都是 crossing 点的时间斜率。斜率太小的时候，forward 和 reverse 的导数都会变得很大，数值上要做保护。

### 31.4 implicit solve 的 reverse 公式

PI driver solve 的 forward sensitivity 是：

$$
\dot{x}=-J_x^{-1}F_P\dot{P}
$$

其中：

$$
x=(t_0,h,c)
$$

$$
P=(C_1,C_2,r_\pi,r_d,S)
$$

reverse 里，如果已经有 $\bar{x}$，不要显式求 $J_x^{-1}$。先解一个 transpose linear system：

$$
J_x^T\mu=\bar{x}
$$

然后把梯度回传到参数 $P$：

$$
\bar{P}
\leftarrow
\bar{P}
-
F_P^T\mu
$$

也就是：

$$
\bar{C}_1,
\bar{C}_2,
\bar{r}_\pi,
\bar{r}_d,
\bar{S}
$$

都会从 implicit solve 里收到贡献。

如果 $J_x$ 不可逆或者病态，这一步也会病态。工程上通常用和 forward sensitivity 同样的 fallback：阻尼 least squares、pseudo-inverse、或者直接标记这个 sample 的解析梯度不可信。

### 31.5 RC tree 的 reverse 公式

如果目标 $Y$ 已经反传到了 root PI 和 sink Elmore，也就是有：

$$
\bar{C}_1,
\bar{C}_2,
\bar{r}_\pi,
\bar{E}_k
$$

先从 PI model 反传到 root moments。因为：

$$
C_1=\frac{N^2}{P},
\qquad
C_2=M-C_1,
\qquad
r_\pi=-\frac{P^2}{N^3}
$$

所以先处理 $C_2=M-C_1$：

$$
\bar{M}
\leftarrow
\bar{M}+\bar{C}_2
$$

$$
\bar{C}_1
\leftarrow
\bar{C}_1-\bar{C}_2
$$

再处理 $C_1=N^2/P$ 和 $r_\pi=-P^2/N^3$：

$$
\bar{N}
\leftarrow
\bar{N}
+
\bar{C}_1\frac{2N}{P}
+
\bar{r}_\pi\frac{3P^2}{N^4}
$$

$$
\bar{P}
\leftarrow
\bar{P}
+
\bar{C}_1\left(-\frac{N^2}{P^2}\right)
+
\bar{r}_\pi\left(-\frac{2P}{N^3}\right)
$$

注意这里的 $P$ 是第三阶 moment，不是上一节参数向量 $P=(C_1,C_2,r_\pi,r_d,S)$。如果担心混淆，代码里建议把第三阶 moment 叫 $P_m$。

Elmore 反传来自：

$$
E_k=
\sum_{v\in path(root,k),v\ne root}
R_vM_v
$$

所以路径上的每个 $v$：

$$
\bar{R}_v
\leftarrow
\bar{R}_v+\bar{E}_kM_v
$$

$$
\bar{M}_v
\leftarrow
\bar{M}_v+\bar{E}_kR_v
$$

然后把 moment recurrence 反传回每条 edge 和每个 node cap。

forward recurrence 是：

$$
M_v=q_v+\sum_uM_u
$$

$$
N_v=\sum_u(N_u-R_uM_u^2)
$$

$$
P_v=\sum_u(P_u-2R_uM_uN_u+R_u^2M_u^3)
$$

reverse 对每个 child $u$ 加上以下贡献。

从 $M_v=q_v+\sum_uM_u$：

$$
\bar{q}_v
\leftarrow
\bar{q}_v+\bar{M}_v
$$

$$
\bar{M}_u
\leftarrow
\bar{M}_u+\bar{M}_v
$$

从 $N_v=\sum_u(N_u-R_uM_u^2)$：

$$
\bar{N}_u
\leftarrow
\bar{N}_u+\bar{N}_v
$$

$$
\bar{R}_u
\leftarrow
\bar{R}_u-\bar{N}_vM_u^2
$$

$$
\bar{M}_u
\leftarrow
\bar{M}_u-2\bar{N}_vR_uM_u
$$

从 $P_v=\sum_u(P_u-2R_uM_uN_u+R_u^2M_u^3)$：

$$
\bar{P}_u
\leftarrow
\bar{P}_u+\bar{P}_v
$$

$$
\bar{R}_u
\leftarrow
\bar{R}_u+\bar{P}_v(-2M_uN_u+2R_uM_u^3)
$$

$$
\bar{M}_u
\leftarrow
\bar{M}_u+\bar{P}_v(-2R_uN_u+3R_u^2M_u^2)
$$

$$
\bar{N}_u
\leftarrow
\bar{N}_u-2\bar{P}_vR_uM_u
$$

最后映射回 segment。

如果 segment $s$ 就是 edge $u$ 的 resistance：

$$
\frac{\partial Y}{\partial R_s}=\bar{R}_u
$$

如果 segment capacitance 一半分到端点 $a,b$：

$$
\frac{\partial Y}{\partial C_s}
=
\frac{1}{2}\bar{q}_a+
\frac{1}{2}\bar{q}_b
$$

### 31.6 所以 timing 时能不能顺便算最终导数

答案分情况：

1. 如果只要少数几个 segment 的 sensitivity：可以在 forward timing 的同时携带 $\dot{x}$，从 RC 往 output 算。

2. 如果要一个 scalar objective 对所有 segment 的梯度：应该 forward timing 一遍存中间量，然后 reverse adjoint 一遍回推。

3. 如果要完整 Jacobian，也就是每个 output 对每个 segment 的导数：那本来就是很大的矩阵，必须多次 forward 或多次 reverse，没有免费的全量结果。

所以你说的“从 cell delay 回推到 RC segment”是 optimizer 里更常用的做法；前面用 dot 写的是 forward sensitivity，是为了先把单个 segment 的解析链路讲清楚。二者数学等价，但计算复杂度不同。

## 32. 最终公式清单和计算方向

这一节把前面的公式压成一个可实现的清单。先说结论：

- 对单个 segment 参数 $\theta_s$，可以用 forward sensitivity，跟着 timing 的中间量一路算 $\dot{x}=dx/d\theta_s$。
- 对一个 scalar 目标 $Y$ 想拿所有 segment 梯度，应该用 reverse adjoint，从 $Y$ 回推到 $R_s,C_s$。
- 普通 timing 只算数值，不会自动得到全量导数；必须额外存中间量，并且额外跑 forward sensitivity 或 reverse adjoint。

### 32.1 单个 segment 的 forward sensitivity 全链路

选一个参数：

$$
\theta_s=R_s
\quad\text{or}\quad
\theta_s=C_s
$$

如果 $\theta_s=R_s$，对 RC tree edge $e$ 的 resistance 种子是：

$$
\dot{R}_e=
\begin{cases}
1,& e=s\\
0,& e\ne s
\end{cases}
$$

如果 $\theta_s=C_s$，并且 segment capacitance 一半分到两个端点 $a,b$，则：

$$
\dot{q}_a=\frac{1}{2},
\qquad
\dot{q}_b=\frac{1}{2}
$$

其他 node cap derivative 为 0。

RC moment 的 bottom-up forward 递推是：

$$
M_v=q_v+\sum_{u\in child(v)}M_u
$$

$$
\dot{M}_v=\dot{q}_v+\sum_{u\in child(v)}\dot{M}_u
$$

$$
N_v=\sum_{u\in child(v)}(N_u-R_uM_u^2)
$$

$$
\dot{N}_v=
\sum_{u\in child(v)}
\left(
\dot{N}_u
-\dot{R}_uM_u^2
-2R_uM_u\dot{M}_u
\right)
$$

$$
P_v=\sum_{u\in child(v)}(P_u-2R_uM_uN_u+R_u^2M_u^3)
$$

$$
\dot{P}_v=
\sum_{u\in child(v)}
\left(
\dot{P}_u
-2\dot{R}_uM_uN_u
-2R_u\dot{M}_uN_u
-2R_uM_u\dot{N}_u
+2R_u\dot{R}_uM_u^3
+3R_u^2M_u^2\dot{M}_u
\right)
$$

sink Elmore 对 sink $k$：

$$
E_k=\sum_{v\in path(root,k),v\ne root}R_vM_v
$$

$$
\dot{E}_k=
\sum_{v\in path(root,k),v\ne root}
\left(
\dot{R}_vM_v+R_v\dot{M}_v
\right)
$$

root PI 参数：

$$
C_1=\frac{N_r^2}{P_r},
\qquad
C_2=M_r-C_1,
\qquad
r_\pi=-\frac{P_r^2}{N_r^3}
$$

$$
\dot{C}_1=
\frac{2N_r\dot{N}_rP_r-N_r^2\dot{P}_r}{P_r^2}
$$

$$
\dot{C}_2=\dot{M}_r-\dot{C}_1
$$

$$
\dot{r}_\pi=
-\frac{2P_r\dot{P}_r}{N_r^3}
+\frac{3P_r^2\dot{N}_r}{N_r^4}
$$

### 32.2 Driver resistance 和 PI solve 的展开

令：

$$
C_{tot}=C_1+C_2
$$

$$
\Delta C=\frac{10^{-15}}{cap\_unit}
$$

$$
d_1=D(S,C_{tot}),
\qquad
d_2=D(S,C_{tot}+\Delta C)
$$

$$
r_d=\kappa |d_1-d_2|,
\qquad
\kappa=-\frac{\ln(v_{th})}{\Delta C}
$$

在 $d_1\ne d_2$ 的可导区间：

$$
\dot{r}_d=
\kappa\,\operatorname{sign}(d_1-d_2)
\left(
\dot{d}_1-\dot{d}_2
\right)
$$

$$
\dot{d}_1=D_S(S,C_{tot})\dot{S}+D_c(S,C_{tot})(\dot{C}_1+\dot{C}_2)
$$

$$
\dot{d}_2=D_S(S,C_{tot}+\Delta C)\dot{S}
+D_c(S,C_{tot}+\Delta C)(\dot{C}_1+\dot{C}_2)
$$

PI solve 仍然写成：

$$
F(x,P_{drv})=0
$$

其中：

$$
x=(t_0,h,c)^T
$$

$$
P_{drv}=(C_1,C_2,r_\pi,r_d,S)^T
$$

所以：

$$
\dot{x}
=
-J_x^{-1}
\left(
F_{C_1}\dot{C}_1
+F_{C_2}\dot{C}_2
+F_{r_\pi}\dot{r}_\pi
+F_{r_d}\dot{r}_d
+F_S\dot{S}
\right)
$$

这里 $c$ 是 PI solve 里的 effective capacitance，不是某个 segment capacitance。

### 32.3 Waveform coefficient 的 forward derivative 全展开

定义：

$$
z_1=\frac{1}{r_\pi C_1},
\qquad
k_0=\frac{1}{r_dC_2}
$$

$$
a=r_\pi r_dC_1C_2,
\qquad
b=r_d(C_1+C_2)+r_\pi C_1
$$

$$
g=\sqrt{b^2-4a}
$$

$$
p_1=\frac{b+g}{2a},
\qquad
p_2=\frac{b-g}{2a}
$$

令：

$$
H=p_1p_2
$$

则：

$$
\dot{z}_1=-z_1\left(\frac{\dot{r}_\pi}{r_\pi}+\frac{\dot{C}_1}{C_1}\right)
$$

$$
\dot{k}_0=-k_0\left(\frac{\dot{r}_d}{r_d}+\frac{\dot{C}_2}{C_2}\right)
$$

$$
\dot{a}=a\left(
\frac{\dot{r}_\pi}{r_\pi}
+\frac{\dot{r}_d}{r_d}
+\frac{\dot{C}_1}{C_1}
+\frac{\dot{C}_2}{C_2}
\right)
$$

$$
\dot{b}=\dot{r}_d(C_1+C_2)+r_d(\dot{C}_1+\dot{C}_2)+\dot{r}_\pi C_1+r_\pi\dot{C}_1
$$

$$
\dot{g}=\frac{2b\dot{b}-4\dot{a}}{2g}
$$

$$
\dot{p}_1=\frac{\dot{b}+\dot{g}}{2a}-p_1\frac{\dot{a}}{a}
$$

$$
\dot{p}_2=\frac{\dot{b}-\dot{g}}{2a}-p_2\frac{\dot{a}}{a}
$$

$$
\dot{H}=p_2\dot{p}_1+p_1\dot{p}_2
$$

$$
k_2=\frac{z_1}{H}
$$

$$
\dot{k}_2=\frac{\dot{z}_1}{H}-k_2\frac{\dot{H}}{H}
$$

$$
n_1=1-k_2(p_1+p_2)
$$

$$
\dot{n}_1=-\dot{k}_2(p_1+p_2)-k_2(\dot{p}_1+\dot{p}_2)
$$

$$
k_1=\frac{n_1}{H}
$$

$$
\dot{k}_1=\frac{\dot{n}_1}{H}-k_1\frac{\dot{H}}{H}
$$

$$
n_4=k_1p_1+k_2,
\qquad
D_4=p_2-p_1
$$

$$
\dot{n}_4=\dot{k}_1p_1+k_1\dot{p}_1+\dot{k}_2
$$

$$
\dot{D}_4=\dot{p}_2-\dot{p}_1
$$

$$
k_4=\frac{n_4}{D_4}
$$

$$
\dot{k}_4=\frac{\dot{n}_4}{D_4}-k_4\frac{\dot{D}_4}{D_4}
$$

$$
k_3=-k_1-k_4
$$

$$
\dot{k}_3=-\dot{k}_1-\dot{k}_4
$$

这组公式给出 $w=(t_0,h,k_0,k_1,k_2,k_3,k_4,p_1,p_2)$ 的全部 forward derivative。

### 32.4 最终 timing output 的 forward 解析式

driver crossing：

$$
\dot{T}_\alpha=-\frac{V_{o,w}^{(\alpha)}\dot{w}}{V_{o,T}^{(\alpha)}}
$$

load crossing：

$$
\dot{L}_\alpha=
-\frac{V_{l,w}^{(\alpha)}\dot{w}+V_{l,E}^{(\alpha)}\dot{E}_k}{V_{l,L}^{(\alpha)}}
$$

cell delay：

$$
\dot{d}_{cell}=D_S\dot{S}+D_c\dot{c}
$$

cell output slew：

$$
\dot{s}_{cell}=\frac{\dot{T}_{v_h}-\dot{T}_{v_l}}{derate}
$$

raw net delay：

$$
\dot{d}_{net,k}^{raw}=\dot{L}_{v_{th}}-\dot{T}_{v_{th}}
$$

raw net slew：

$$
\dot{s}_{net,k}^{raw}=\frac{\dot{L}_{v_h}-\dot{L}_{v_l}}{driver\_derate}
$$

threshold adjusted net output：

$$
\dot{d}_{net,k}=\dot{d}_{net,k}^{raw}+sign\cdot\beta\cdot\dot{s}_{net,k}^{raw}
$$

$$
\dot{s}_{net,k}=\gamma\dot{s}_{net,k}^{raw}
$$

这就是单个 $R_s$ 或 $C_s$ 对 cell delay、cell slew、net sink delay、net sink slew 的最终解析式。

### 32.5 从 timing output 回推到所有 segment 的 reverse 入口

如果目标是 cell delay：

$$
Y=d_{cell}=D(S,c)
$$

初始化：

$$
\bar{S}\leftarrow\bar{S}+D_S,
\qquad
\bar{c}\leftarrow\bar{c}+D_c
$$

这里的 $\bar{c}$ 是对 PI solve unknown $x=(t_0,h,c)$ 第三个分量的 adjoint。然后用 31.4 的 implicit reverse 把它回推到 $C_1,C_2,r_\pi,r_d,S$。

如果目标是 cell output slew：

$$
Y=s_{cell}=\frac{T_{v_h}-T_{v_l}}{derate}
$$

初始化：

$$
\bar{T}_{v_h}\leftarrow\bar{T}_{v_h}+\frac{1}{derate},
\qquad
\bar{T}_{v_l}\leftarrow\bar{T}_{v_l}-\frac{1}{derate}
$$

再用 31.3 的 driver crossing reverse 回推到 $w$。

如果目标是 threshold adjusted net delay：

$$
Y=d_{net,k}=d_{net,k}^{raw}+sign\cdot\beta\cdot s_{net,k}^{raw}
$$

初始化：

$$
\bar{d}_{net,k}^{raw}\leftarrow\bar{d}_{net,k}^{raw}+1
$$

$$
\bar{s}_{net,k}^{raw}\leftarrow\bar{s}_{net,k}^{raw}+sign\cdot\beta
$$

再由：

$$
d_{net,k}^{raw}=L_{v_{th}}-T_{v_{th}}
$$

得到：

$$
\bar{L}_{v_{th}}\leftarrow\bar{L}_{v_{th}}+\bar{d}_{net,k}^{raw}
$$

$$
\bar{T}_{v_{th}}\leftarrow\bar{T}_{v_{th}}-\bar{d}_{net,k}^{raw}
$$

由：

$$
s_{net,k}^{raw}=\frac{L_{v_h}-L_{v_l}}{driver\_derate}
$$

得到：

$$
\bar{L}_{v_h}\leftarrow\bar{L}_{v_h}+\frac{\bar{s}_{net,k}^{raw}}{driver\_derate}
$$

$$
\bar{L}_{v_l}\leftarrow\bar{L}_{v_l}-\frac{\bar{s}_{net,k}^{raw}}{driver\_derate}
$$

之后使用 31.3、31.4、31.5 一路回推，最终得到：

$$
\frac{\partial Y}{\partial R_s}=\bar{R}_s
$$

$$
\frac{\partial Y}{\partial C_s}=\frac{1}{2}\bar{q}_a+\frac{1}{2}\bar{q}_b
$$

这就是“从 cell delay / net delay 回推到 RC segment”的做法。

### 32.6 为什么前面会出现 bottom-up

RC moment 本身就是从 leaf/sink 往 root 汇总的量，所以无论 timing 还是 forward sensitivity，$M,N,P$ 都天然是 bottom-up 计算。

但这不等于 optimizer 的梯度方向必须 bottom-up：

- forward sensitivity：先选一个 segment，从它对 root moment、driver waveform、sink crossing 的影响一路推到 output。
- reverse adjoint：先选一个 output 或 scalar loss，从 output 反推到 waveform、PI solve、root moment、每个 segment。

所以两件事不要混在一起：

1. RC moment 数值计算是 bottom-up；
2. 对所有 segment 求一个目标函数梯度，推荐 reverse adjoint；
3. 普通 timing forward 只能顺便保存 reverse 需要的中间量，不能不加额外计算就得到最终全量导数。

## 33. 面向 routing 的全 segment 梯度方案

如果目标是指导 routing，不建议求完整 Jacobian：

$$
\frac{\partial(d_{cell},s_{cell},d_{net},s_{net})}{\partial(R_s,C_s)}
$$

这个矩阵太大。真正该做的是先定义一个 scalar routing objective：

$$
Y=Y(AT, RAT, slew, slack)
$$

然后求：

$$
\frac{\partial Y}{\partial R_s},
\qquad
\frac{\partial Y}{\partial C_s}
\qquad
\forall s\in routing\ segments
$$

这时应该用 reverse adjoint，而不是对每个 segment 跑一遍 forward sensitivity。

### 33.1 推荐目标函数

硬 WNS 只会把梯度打到最差 endpoint 上，太稀疏。routing 更适合用 smooth 或 weighted objective，例如：

$$
Y=\sum_i w_i\,\operatorname{softplus}\left(\frac{AT_i-RAT_i}{\tau}\right)\tau
$$

或者简化成 criticality weighted delay：

$$
Y=\sum_i crit_i\,AT_i
$$

如果用 setup slack：

$$
slack_i=RAT_i-AT_i
$$

那么对 timing degradation 的目标可以写成：

$$
Y=\sum_i w_i\max(0,-slack_i)
$$

实际实现时硬 $\max$ 可以用 active subgradient，smooth 版本更稳定。

### 33.2 总体流程

一次 routing iteration 里做：

1. 正常 DMP forward timing，得到 $AT,slew,slack$。
2. 保存或可重算所有 reverse 需要的中间量。
3. 从 scalar objective $Y$ 初始化 endpoint adjoint。
4. 按 timing level 反向传播 adjoint。
5. 把 net sink delay/slew 的 adjoint 累加到每个 net 的：

$$
\bar{C}_1,\bar{C}_2,\bar{r}_\pi,\bar{E}_k
$$

6. 对每个 RC tree 做 reverse pass，得到：

$$
\bar{R}_s=\frac{\partial Y}{\partial R_s}
$$

$$
\bar{C}_s=\frac{\partial Y}{\partial C_s}
$$

7. 转成 router cost。

如果 segment 的物理变量是长度 $l_s$，并且：

$$
R_s=r_{unit}l_s,
\qquad
C_s=c_{unit}l_s
$$

则 timing cost 对长度的局部线性梯度是：

$$
\frac{\partial Y}{\partial l_s}
=
\bar{R}_sr_{unit}+\bar{C}_sc_{unit}
$$

router 可以把它作为该 routing edge 的 timing cost。

### 33.3 需要新增或保留的数据

当前 DMP forward 已经有很多需要的量：

$$
y1,y2,y3
$$

对应 root moments：

$$
M,N,P
$$

还有：

$$
down\_cap,
node\_delay,
elmore\_delay,
C_1,C_2,r_\pi
$$

为了 reverse 到 routing segment，还需要保留或新增：

- `node_order`
- `parent_node`
- `res_parent`
- `node_cap`
- `parent_edge_id`，也就是每个 RC node 的 parent edge 对应哪个 routing segment
- segment capacitance 如何分摊到 node cap 的映射
- timing winner 信息，例如当前 active arc、active attr、active transition
- PI solve 的中间量，或者保证 reverse 时能从 forward 状态重算出来

注意当前代码的 `release_dmp_rc_build_only_fields` 可能释放部分 RC build graph。做 gradient 时不能释放 reverse 必需字段。

### 33.4 Reverse timing 的 CUDA 结构

当前 `dmpBackwardKernel` 是 RAT propagation，不是 gradient propagation。需要新增一套 adjoint kernels。

forward timing 是按 level 从 source 到 sink：

$$
level=0,1,2,\dots
$$

reverse gradient 要反过来：

$$
level=L,L-1,\dots,0
$$

每个 reverse level 做三类事情：

1. pin winner reverse：把 pin output adjoint 分发回 winning candidate。
2. net delay/slew reverse：把 sink delay/slew adjoint 回传到 driver waveform 和 sink Elmore。
3. gate delay/slew reverse：把 gate output adjoint 回传到 input slew、load PI 参数、Liberty LUT 输入。

如果继续使用 hard winner，那么 winner 以外的 candidate 梯度为 0。这是 active subgradient：

$$
\bar{x}_{winner}\leftarrow\bar{x}_{winner}+\bar{y}
$$

如果想让 router 更平滑，可以把 hard max 改成 softmax/softmin：

$$
y=\tau\log\sum_i e^{x_i/\tau}
$$

则：

$$
\bar{x}_i\leftarrow\bar{x}_i+\bar{y}\frac{e^{x_i/\tau}}{\sum_j e^{x_j/\tau}}
$$

softmax 会增加计算量和存储量，但梯度更密。

### 33.5 RC tree reverse 可以做到线性复杂度

不要对每个 sink 展开整条 path，否则可能变成：

$$
O\left(\sum_k path\_len(k)\right)
$$

正确做法是把 Elmore forward 看成 top-down recurrence：

$$
D_r=0
$$

$$
D_v=D_{parent(v)}+R_vM_v
$$

pin node 的：

$$
E_k=D_v
$$

reverse 时先初始化 pin node：

$$
\bar{D}_v\leftarrow\bar{D}_v+\bar{E}_k
$$

然后按 node order 反向，也就是 leaf 到 root：

$$
\bar{D}_{parent(v)}\leftarrow\bar{D}_{parent(v)}+\bar{D}_v
$$

$$
\bar{R}_v\leftarrow\bar{R}_v+\bar{D}_vM_v
$$

$$
\bar{M}_v\leftarrow\bar{M}_v+\bar{D}_vR_v
$$

这样 Elmore reverse 是：

$$
O(V_{net})
$$

不是按 sink path 数重复走。

root PI 反传之后，再对 moment recurrence 做 root 到 leaf 的 reverse pass，得到每条 edge 的 $\bar{R}$ 和每个 node cap 的 $\bar{q}$。最后由 segment cap 分摊关系得到：

$$
\frac{\partial Y}{\partial C_s}
=
\frac{1}{2}\bar{q}_a+\frac{1}{2}\bar{q}_b
$$

### 33.6 时间复杂度

定义：

- $N$：net 数
- $V$：RC node 总数
- $E$：routing segment / RC edge 总数
- $P$：pin 总数
- $A$：timing arc candidate 总数
- $K$：attr 数，当前基本是常数，约等于 `NUM_ATTR`

一次普通 DMP forward 的主要复杂度近似是：

$$
T_{forward}=O(K(V+E)+K A_{active})
$$

其中 $A_{active}$ 包括 gate arc、net arc、gate-net pair lane 这些实际 timing candidate。

如果求一个 scalar objective 对所有 segment 的梯度，reverse-mode 复杂度是：

$$
T_{reverse}=O(K(V+E)+K A_{active})
$$

所以总复杂度仍然是线性的：

$$
T_{grad}=O(K(V+E)+K A_{active})
$$

常数大约是普通 timing 的 2 到 4 倍，取决于 reverse 时是重算 PI/crossing 中间量，还是 forward 存下来。

关键点是：

$$
T_{grad}\ne O(E\cdot T_{forward})
$$

如果对每个 segment 单独跑 forward sensitivity，才会变成：

$$
O(E\cdot (K(V+E)+KA_{active}))
$$

这对 routing segment 全量梯度基本不可用。

如果你要完整 Jacobian，也就是每个 endpoint output 对每个 segment：

$$
J_{i,s}=\frac{\partial output_i}{\partial segment_s}
$$

那矩阵本身大小就是：

$$
O(P E)
$$

无论 forward 还是 reverse 都很贵，不适合作为 router 每轮内循环。

### 33.7 空间复杂度

普通 forward 已有 RC/timing 状态：

$$
O(KV+KP+KE+KA_{scratch})
$$

做 reverse 需要新增 adjoint：

$$
\bar{AT},\bar{slew},\bar{C}_1,\bar{C}_2,\bar{r}_\pi,\bar{E}
$$

以及 RC adjoint：

$$
\bar{M},\bar{N},\bar{P},\bar{D},\bar{q},\bar{R}
$$

空间近似：

$$
S_{reverse}=O(KV+KP+KE)
$$

如果把每个 gate-net candidate 的 PI solve 中间量全部存下来，空间会增加：

$$
O(KA_{active})
$$

更省显存的方案是 reverse 时重算局部 PI solve / crossing slope，只存 winner 和必要 input/output 值。这样时间增加，空间减少。

### 33.8 CUDA 并行策略

推荐分成四组 kernel。

第一组：forward timing，不变，但要保留 reverse 需要的状态。

第二组：objective init kernel。每个 endpoint/attr 一个线程，写：

$$
\bar{AT},\bar{slew},\bar{RAT}
$$

第三组：reverse timing kernels。按 level 反向 launch。每个 arc、net sink candidate 或 gate-net lane 一个 thread 或一组 threads。梯度累加用 `atomicAdd`。

第四组：reverse RC kernels。可以沿用当前 RC propagate 的并行粒度：

$$
block/thread \approx net \times attr
$$

每个 net 内部按 `node_order` 做 serial tree pass。这样实现简单，复杂度仍是 $O(KV)$，并且跨 net 并行度很高。

如果某些 high-fanout net 太大，可以再做优化：按 tree depth 分层，edge parallel 做 reduction。但第一版不建议这么做，因为 launch 和同步复杂度会高很多。

### 33.9 数值和非光滑点

以下地方都是 piecewise / non-smooth：

- max/min arrival winner
- WNS/TNS hard selection
- Liberty table interpolation cell 边界
- PI/CAP/ZERO_C2 algorithm branch
- $r_d=\kappa|d_1-d_2|$ 的 absolute value
- crossing slope 很小
- implicit solve 的 $J_x$ 病态或不可逆

工程策略：

1. 默认用 active subgradient。
2. 对 routing objective 用 smooth criticality，减少 winner 抖动。
3. 对 crossing slope 和 $J_x$ condition 做保护。
4. 梯度异常时 fallback 到 cap model、finite difference check 或直接 clip。
5. router 用梯度做局部 cost，不要把它当全局精确预测。

### 33.10 最推荐的实现路线

第一版应该这样做：

1. 只支持 PI branch 的 active subgradient。
2. 目标函数先用 criticality weighted arrival 或 smooth TNS。
3. reverse timing 只沿当前 winning arcs 回传。
4. reverse RC 做线性 pass，输出每个 routing segment 的 $\bar{R}_s,\bar{C}_s$。
5. router edge cost 用：

$$
cost_s=base_s+\lambda_t\left(\bar{R}_sr_{unit}+\bar{C}_sc_{unit}\right)
$$

6. 做 finite difference 抽样验证：随机选几十个 segment，扰动 $R_s,C_s$，比较：

$$
\Delta Y_{pred}=\bar{R}_s\Delta R_s+\bar{C}_s\Delta C_s
$$

和实际 rerun timing 的 $\Delta Y$。

如果这条链路通过，再考虑 softmax winner、多 endpoint objective、多 path adjoint 和高 fanout RC parallelization。

## 34. 为什么不需要显式构造完整 Jacobian

这里说“不要求 Jacobian”，准确含义是：

**不要求显式构造完整 Jacobian 矩阵。**

不是说不求导数。

设所有 timing output 组成向量：

$$
y=f(\theta)
$$

其中：

$$
\theta=(R_1,C_1,R_2,C_2,\dots)
$$

是所有 routing segment 的参数。

完整 Jacobian 是：

$$
J=\frac{\partial y}{\partial \theta}
$$

其中每个元素是：

$$
J_{i,s}=\frac{\partial y_i}{\partial \theta_s}
$$

如果有 $m$ 个 timing output、$n$ 个 segment 参数，那么：

$$
J\in\mathbb{R}^{m\times n}
$$

这个矩阵本身就很大。对 routing 来说，通常也不直接需要每个 output 对每个 segment 的单独导数。

真正要指导 routing 的，一般是一个 scalar objective：

$$
Y=g(y)
$$

例如：

$$
Y=\sum_i crit_i\,AT_i
$$

或者 smooth TNS / slack penalty。

router 需要的是：

$$
\nabla_\theta Y
=
\frac{\partial Y}{\partial \theta}
$$

也就是：

$$
\frac{\partial Y}{\partial R_s},
\qquad
\frac{\partial Y}{\partial C_s}
\qquad
\forall s
$$

链式法则给出：

$$
\frac{\partial Y}{\partial \theta}
=
\left(\frac{\partial y}{\partial \theta}\right)^T
\frac{\partial Y}{\partial y}
$$

也就是：

$$
\nabla_\theta Y
=
J^T\lambda
$$

其中：

$$
\lambda=\nabla_y Y
$$

关键点是：

$$
\text{需要的是 }J^T\lambda
\quad
\text{不是完整的 }J
$$

reverse adjoint 正是在不显式形成 $J$ 的情况下，直接计算：

$$
J^T\lambda
$$

所以它仍然是在求导数，只是求的是 scalar objective 对所有 segment 的梯度。

举例，如果 routing objective 是：

$$
Y=\sum_i w_i d_i
$$

那么某个 segment resistance 的梯度是：

$$
\frac{\partial Y}{\partial R_s}
=
\sum_i w_i
\frac{\partial d_i}{\partial R_s}
$$

显式 Jacobian 会把所有：

$$
\frac{\partial d_i}{\partial R_s}
$$

都单独存下来。

reverse adjoint 不这么做。它从 $Y$ 开始，把每个 output 的权重 $w_i$ 作为 adjoint 往回传，沿 timing graph 和 RC tree 自动累加所有贡献，最后直接得到：

$$
\frac{\partial Y}{\partial R_s}
$$

和：

$$
\frac{\partial Y}{\partial C_s}
$$

所以：

1. 如果要“每个 output 对每个 segment 的导数”，那确实需要完整 Jacobian 或等价的大矩阵。
2. 如果要“一个 routing cost 对所有 segment 的导数”，只需要 gradient。
3. reverse-mode 的输出就是这个 gradient，数学上等价于计算 $J^T\lambda$。

因此全量 routing segment 梯度不是不求 Jacobian 意义下的导数，而是避免显式构造巨大 Jacobian，只计算 router 真正需要的 Jacobian-vector product。

