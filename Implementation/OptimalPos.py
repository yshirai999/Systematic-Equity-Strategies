import dsp
import cvxpy as cp
import numpy as np
from Models import BG
from Distortions import minmaxvar as mmv
import matplotlib.pyplot as plt

params = [0.04,10/52,1,5/52,0.02,10/52,1.5,5/52]

benchmark = False

k = 8
N = 50
a = np.linspace(0.5,2,N)
W = 1
M = [-0.5,0.5]
model = BG(k,M,params)
p = model.p
q = model.q
x = model.y

# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 1, 1])
# axes.set_xlim(M[0], M[1])
# axes.set_ylim(0, max([max(p),max(q)]))
# axes.plot(x, p)
# axes.plot(x, q)
# plt.show()

lam = 0.25
dist = mmv(lam)
Phi = dist.Phi(a)

theta = 0.75
alpha = 1.25
beta = 0.25
P = np.diag(p)

y = cp.Variable(2**k)
z = cp.Variable(2**k)

if benchmark:
    f = dsp.inner(z, P @ (y - cp.multiply(W,cp.exp(x))))
    rho = p @ (cp.multiply(theta, cp.power(z,alpha))+cp.multiply(1-theta,cp.power(z,-beta)))
    obj = dsp.MinimizeMaximize(rho+f)
    constraints = [q @ y == W, p @ z == 1, z >= 0]
    for i in range(N): #range(len(a)):
        constraints.append(p @ cp.maximum(z-a[i],0) <= Phi[i])
else:
    f = dsp.inner(z, P @ (y - q @ y))
    rho = p @ (cp.multiply(theta, cp.power(z,alpha))+cp.multiply(1-theta,cp.power(z,-beta)))
    obj = dsp.MinimizeMaximize(rho+f)
    constraints = [p @ z == 1, z >= 0]
    for i in range(N): #range(len(a)):
        constraints.append(p @ cp.maximum(z-a[i],0) <= Phi[i])

prob = dsp.SaddlePointProblem(obj, constraints)
prob.solve()  # solves the problem

print(prob.value)

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.75, 0.75])
M = [-0.3,0.3]
axes.set_xlim(np.log(W)+M[0], np.log(W)+M[-1])
#axes.set_ylim(min(y.value), max(y.value))
#axes.plot(x,y.value-W*np.exp(x)
axes.plot(x,y.value)
if benchmark:
    axes.plot(x,W*np.exp(x))
    plt.savefig('PosBench.png')
    plt.show()
else:
    plt.savefig('Pos.png')
    plt.show()
    

# Constraint satisfied
print(q @ y.value)
zz = z.value
N = len(a)
const = []
for i in range(N): #range(len(a)):
    const.append(p @ np.maximum(zz-a[i],0))

const = np.array(const)
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.75, 0.75])
axes.set_xlim(a[0], a[-1])
axes.set_ylim(min(const), max(Phi))
axes.plot(a,const)
axes.plot(a,Phi)
# axes.plot(a,dist.Psi(a))
plt.show()