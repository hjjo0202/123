# Feedback Control 1
---
1. Open-loop control
- 되돌아오지 않고 가는 제어형식. 입력을 하고난 순간 그 이후는 사용자가 건드릴 수 없다.
- Open-loop일 경우 Actuator와 Process가 내 기대와 동일하게 작동해야 정상적으로 작동이 된다.

2. Closed-loop control
- Open loop와 다르게 Feedback을 사용하여 내가 기대하는 출력값과 실제 출력값의 차이(error)를 Feedback으로 다시 재반영함으로써 내 기대값과 동일한 값을 얻을 수 있다.

3. Error signal  
$$Y(s)= \frac{G_c(s)G(s)}{1+G_c(s)G(s)}R(s)+\frac{G(s)}{1+G_c(s)G(s)}-\frac{G_c(s)G(s)}{1+G_c(s)G(s)}$$  
$$E(s)=\frac{1}{1+G_c(s)G(s)}R(s)-\frac{G(s)}{1+G_c(s)G(s)}T_d(s)+\frac{G_c(s)G(s)}{1+G_c(s)G(s)}N(s)$$  
$$L(s)=G_c(s)G(s)$$
$$E(s)=\frac{1}{1+L(s)}R(s)-\frac{G(s)}{1+L(s)}T_d(s)+\frac{L(s)}{1+L(s)}N(s)$$  
$$S(s)=\frac{1}{1+L(s)}$$
$$C(s)=\frac{L(s)}{1+L(s)}$$  
$$E(s)=S(s)R(s)-S(s)G(s)T_d(s)+C(s)N(s)$$  

4. Trade-off in error signal  
-Disturbance를 최소화, Noise를 최소화  
$$E(s)=S(s)R(s)-S(s)G(s)T_d(s)+C(s)N(s)$$
$S(s)$ -> 0 이라면 Disturbance 최소화  
$C(s)$ -> 0 이라면 Noise 최소화  
Disturbance는 Low Frequency, Noise는 High Frequency이기 때문에 접점을 잘 찾으면 이 둘을 한번에 줄일 수 있는 지점을 찾을 수 있다.

5. Variation on $G(s)$  
$$\Delta E(s)=\frac{-G_c(s)\Delta G(s)}{[1+G_c(s)G(s)+G_c(s)\Delta G(s)][1+G_c(s)G(s)]}R(s)$$  
만약 $\Delta G(s)$가 작으면  

$$ \Delta E(s)=-\frac{\Delta G(s)}{L(s)G(s)}R(s)$$  

---
#Feedback Control 2
---
1. DC Motor example  
$V_a(s)$를 입력으로 넣었을 때 출력이 $w(s)$인 Motor System
-입력: $V_a(s), L_a, R_a$를 통과하는 전류 $i_a(s)$
- 전류 값은 $K_m$(모터특성)에 비례하여 $T_m(s)$로 변환된다.
- $\frac{1}{Js+b}$: 주파수가 커질수록 0에 가까워짐. 시스템 구동 초반에는 모터에 큰 저항을 주지만 시간이 지날수록 작아진다.
- $T_d(s)$:Load Torque Disturbance. 주변 환경에 의한 불규칙적 요인. 혹은 내부 요인
- $K_b$:역기전력. 목표 속도 도달까지 시간을 지연시킴  

1-1. Disturbance에 의한 Error
$$E(s)=S(s)R(s)-S(s)G(s)T_d(s)+C(s)N(s)$$  
위 식에서 $R(s)=0, N(s)=0$이면   
$$E(s)=-\frac{G(s)}{1+L(s)}T_d(s)$$  
이때 $G(s)=\frac{1}{Js+b}, K_b=\frac{K_m}{R_a}\frac{1}{Js+b}$이다. 따라서  
$$E(s)=\frac{1}{Js+b+\frac{K_mK_b}{R_a}}T_d(s)$$
여기서 $T_d(s)$에 곱해진 것이 disturbance에 의한 Error가 된다.    
$T_d(s)=\frac{D}{s}$ 라 하자.  
$\frac{1}{s}$은 시간축에서의 Unit Step Function이었다. 이를 주파수 영역으로 가져온 것이다.  
Final value theorem 적용 시
$$\lim_{t \rightarrow \infty}e(t)=\lim_{s \rightarrow 0}sE(s)=\frac{D}{b+\frac{K_mK_b}{R_a}}$$  

$$G_1(s)=K_a\frac{K_m}{R_a}, H(s)=K_t+\frac{K_b}{K_a},G_2(s)=\frac{1}{Js+b}$$
$$R(s)=0, N(s)=0$$  
$$E(s)=\frac{G_2(s)}{1+G_1(s)G_2(s)H(s)}T_d(s)=\frac{1}{Js+b+\frac{K_m}{R_a}(K_tK_a+K_b)}$$  
$$K_a>>K_b, K_a>>b → E(s)=\frac{1}{Js+\frac{K_mK_tK_a}{R_a}}$$  
- Final value theorem을 진행
$$\lim_{t \rightarrow \infty}e(t)=\lim_{s \rightarrow 0}sE(s)=\frac{DR_a}{K_mK_tK_a}$$  
-분모에 $K_a$가 존재하므로 정량적으로 존재하던 error를 줄일 수 있다.
-Feedback을 통해 disturbance에 의한 error를 감소
-Loop gain을 키울수록 disturbance에 의한 error를 감소  

1-2. Noise에 의한 error
$R(s)=0,T_d(s)=0,C(s)N(s)$ 식으로 동일하게 진행한다.  

따라서 Closed Loop는 feedback을 통해 error를 줄이고, 추가로 gain을 늘리면 disturbance가 감소하고, gain을 줄이면 noise가 감소한다.

```
## 매트랩 코드

#LT 주어진 함수
clear all;

% Symbolic variable declarations
syms t s
syms a positive % a는 항상 양수

% Compute Laplace transforms here
% laplace(원본함수, 원본함수 변수: t, LT 변수: s)란 뜻
Fs_a = laplace(1, t, s) 
Fs_b = laplace(exp(-a*t), t, s)
Fs_c = laplace(t, t, s)
Fs_d = laplace(dirac(t-a), t, s)
Fs_e = laplace(cos(t), t, s)
Fs_f = laplace(t^2*exp(-a*t), t, s)
Fs_g = laplace(exp(-a*t)*sin(t), t, s)
```
```
##위 함수의 미분/적분/Exp/Subs
clear all;

syms t tau f(t)
syms a positive

dfdt = diff(f(t))
intf = int(subs(f(t), t, tau), tau, [0 t])
fsft = exp(a*t)*f(t)
tsft = subs(f(t), t, t-a)*heaviside(t-a)

Fs_dfdt = simplify(laplace(dfdt))
Fs_intf = simplify(laplace(intf))
Fs_fsft = simplify(laplace(fsft))
Fs_tsft = simplify(laplace(tsft))

syms F(s)

Fs_dfdt = subs(Fs_dfdt, laplace(f(t), t, s), F(s))
Fs_intf = subs(Fs_intf, laplace(f(t), t, s), F(s))
Fs_fsft = subs(Fs_fsft, laplace(f(t), t, s), F(s - a))
Fs_tsft = subs(Fs_tsft, laplace(f(t), t, s), F(s))
```
```
## LT example
clear all;

% Define X(s)
syms x(t) X t s
m = 1;
c = 2;
k = 10;
F = 1;

eqn = m*diff(x(t), 2) + c*diff(x(t)) + k*x(t) == F
L_eqn = laplace(eqn)
L_eqn_final = subs(L_eqn, {laplace(x(t), t, s), x(0), subs(diff(x(t)), t, 0)}, {X, 0, 0})
X = solve(L_eqn_final, X)
syms x
x = ilaplace(X)
t_ = linspace(0,5,150);
plot(t_, subs(x, t, t_) )
```
```
## TF
clear all;

% Define X(s)
syms x(t) u(t) X U t s
m = 1;
c = 2;
k = 10;

eqn = m*diff(x(t), 2) + c*diff(x(t)) + k*x(t) == u(t)
L_eqn = laplace(eqn)
L_eqn_final = subs(L_eqn, {laplace(x(t), t, s), laplace(u(t), t, s), x(0), subs(diff(x(t)), t, 0)}, {X, U, 0, 0})
X = solve(L_eqn_final, X)
TF = X / U

% TF를 기반으로 Response 구하기
% 분자는 n 분모는 d가 되도록
[n, d] = numden(TF)
a = sym2poly(n)
b = sym2poly(d)
sys = tf(a,b)
step(sys, 5)
```
