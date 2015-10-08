import numpy as np
import matplotlib.pyplot as plt

'''Este script resuelve el oscilador de
Van der Pool, utilizando el metodo de runge-kutta
de orden 3. Se utilizan dos sets de condiciones
iniciales: 1) dy/ds=0, y=0.1 ;2) dy/ds=0 , y=4.
Finalmente grafica (y vs dy/dy) y (y(s) vd s).
'''
K=1.232  #rut 18769232-6

#definimos las funciones a utilizar para RK3
def f_a_integrar(x,v,k=K):
    return [v, -x-k*(x**2-1)*v]

def get_k1(xn,vn, h, f_a_integrar):
    f_eval = f_a_integrar(xn,vn)
    return h*f_eval[0], h*f_eval[1]

def get_k2(xn,vn, h, f_a_integrar):
    k1= get_k1(xn, vn, h, f_a_integrar)
    f_eval= f_a_integrar(xn + k1[0]/2. , vn +k1[1]/2. )
    return h*f_eval[0], h*f_eval[1]

def get_k3(xn,vn, h, f_a_integrar):
    k1= get_k1(xn, vn, h, f_a_integrar)
    k2= get_k2(yn, h, f_a_integrar)
    f_eval= f_a_integrar(xn - k1[0] - 2*k2[0] ,  vn - k1[1] - 2*k2[1])
    return h*f_eval[0], h*f_eval[1]

def rk3_step(xn,vn,h,f_a_integrar):
    k1=get_k1(xn, vn, h, f_a_integrar)
    k2=get_k2(xn, vn, h, f_a_integrar)
    k3=get_k2(xn, vn, h, f_a_integrar)
    x_new=xn+(1/6.)*(k1[0]+4*k2[0]+k3[0])
    v_new=vn+(1/6.)*(k1[1]+4*k2[1]+k3[1])

    return [x_new, v_new]


N_steps = np.int(1e5)
h=20*np.pi/N_steps

#creamos los arrays en que se guardara la informacion
x1= np.zeros(N_steps)
v1= np.zeros(N_steps)
x2= np.zeros(N_steps)
v2= np.zeros(N_steps)
t=np.linspace(0,20*np.pi,N_steps)

fig = plt.figure(1)
fig.clf()

ax1 = fig.add_subplot(211)
ax1.set_xlabel('y')
ax1.set_ylabel('dy/ds')
ax1.set_xlim(-3,4)
plt.title("Trayectoria en espacio (y,dy/ds) oscilador de Van der Pool")

#primeras condiciones iniciales
x1[0]= 0.1
v1[0]= 0
for i in range (1, N_steps):
    x_new, v_new = rk3_step(x1[i-1],v1[i-1],h,f_a_integrar)
    x1[i]= x_new
    v1[i]= v_new
ax1.plot(x1,v1, label="Condiciones iniciales: y=0.1, dy/ds=0", color='r')
plt.legend(loc='lower right', fontsize=10)

ax2 = fig.add_subplot(212)
ax2.set_xlabel('y')
ax2.set_ylabel('dy/ds')

#segundas condiciones iniciales
x2[0]= 4
v2[0]= 0
for i in range (1, N_steps):
    x_new, v_new = rk3_step(x2[i-1],v2[i-1],h,f_a_integrar)
    x2[i]= x_new
    v2[i]= v_new
ax2.plot(x2,v2, label="Condiciones iniciales: y=4, dy/ds=0")

plt.legend(loc='lower right', fontsize=10)
plt.draw()
plt.savefig('figura1.1.png')
plt.show()

####
fig = plt.figure(2)
fig.clf()
ax3=fig.add_subplot(211)
plt.title("y(s) para condiciones iniciales: y=0.1 , dy/ds=0")
plt.plot(t,x1,color='r')
ax3.set_ylabel('y(s)')
ax3.set_ylim(-4,4)

ax4=fig.add_subplot(212)
plt.title("y(s) para condiciones iniciales: y=4 , dy/ds=0")
plt.plot(t,x2)
ax4.set_xlabel('s')
ax4.set_ylabel('y(s)')


plt.draw()
plt.savefig('figura1.2.png')
plt.show()
