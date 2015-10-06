import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

#parametros a usar
sigma= 10
beta= 8/3.
rho= 28

#set de condiciones iniciales
w0=[1,1,1]
t0=0


fig = plt.figure(1)
fig.clf()

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

def f_a_integrar(t,w,s=sigma,r=rho,b=beta):
    x, y, z = w
    return [s*(y-x), x*(r-z)-y, x*y-b*z]

#creamos el resolvedor
r=ode(f_a_integrar)
r.set_integrator('dopri5')
r.set_initial_value(w0,t0)

#tiempo conveniente
t=np.linspace(t0,10,100)
#vectores de info a guardar
x=np.zeros(len(t))
y=np.zeros(len(t))
z=np.zeros(len(t))


for i in range(len(t)):
    r.integrate(t[i])
    x[i], y[i], z[i] = r.y

ax.plot(x, y, z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
