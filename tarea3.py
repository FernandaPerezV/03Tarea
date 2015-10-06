import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(1)
fig.clf()

ax1 = fig.add_subplot(111)
ax1.set_xlabel('x')
ax1.set_ylabel('v')


K=1.232  #rut 18769232-6

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
x= np.zeros(N_steps)
v= np.zeros(N_steps)

x[0]= 0.1
v[0]= 0

for i in range (1, N_steps):
    x_new, v_new = rk3_step(x[i-1],v[i-1],h,f_a_integrar)
    x[i]= x_new
    v[i]= v_new

ax1.plot(x,v)

plt.draw()
plt.savefig('figura1.png')
plt.show()
