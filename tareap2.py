import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

'''Este scrip resuelve el sistema de Lorenz para
el caso en que sigma=10, beta=8/3, rho=20 (atractor de lorenz).
Se utiliza runge-kutta de orden 4 con metodo dopri5
de la libreria scipy.integrate.
Utiliza condiciones iniciales x0=1,y0=1,z0=1.
Finalmente grafica x(t),y(t),z(t) en 3D.
'''
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
ax.set_aspect('auto')

#definimos la funcion a integrar
def f_a_integrar(t,w,s=sigma,r=rho,b=beta):
    x, y, z = w
    return [s*(y-x), x*(r-z)-y, x*y-b*z]

#creamos el resolvedor
r=ode(f_a_integrar)
r.set_integrator('dopri5')
r.set_initial_value(w0,t0)

#tiempo conveniente
t=np.linspace(t0,100,10000)

#vectores para guardar la informacion
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
plt.title("Atractor de Lorenz")

plt.draw()
plt.show()
plt.savefig('figura2.png')
