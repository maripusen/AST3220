import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

"Defining a few constants:"
mx = 1000           #GeV
g = 2               #degrees of freedom
gstar = 106.75      #degrees of freedom
gstars = 106.75     #degrees of freedom

cross1 = 10**(-9)   #Cross section for the reaction rate
cross2 = 10**(-10)  #Cross section for the reaction rate
cross3 = 10**(-11)  #Cross section for the reaction rate

def yeq(x,cross):
    return np.log(4.675e17 * g* (x**(3/2))*mx *cross*(np.exp(-x))/np.sqrt(gstar))

def lamb(cross):
    return 2*np.pi*np.sqrt(90)/45 *(gstars)/(np.sqrt(gstar))*(mx)/(np.sqrt(8*np.pi))

def solve(x,cross):
    def dwdx(x,w):
        weq = yeq(x,cross)
        return (np.exp(2*weq-w)-np.exp(w))/x**2
    w0 = yeq(x[0],cross)
    xs = (x[0],x[-1])
    t = np.linspace(xs[0],xs[1],10000)

    solved =solve_ivp(dwdx,xs,[w0], method = "Radau",t_eval = t)
    return solved, yeq(t,cross)

solve1 = solve([1,745],cross1)
solve2 = solve([1,745],cross2)
solve3 = solve([1,745],cross3)

mpl.rcParams['font.size']=14

plt.figure()
plt.subplot(1,3,1)
plt.yscale("log")
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-9")
plt.plot(solve1[0].t,np.exp(solve1[0].y[0]),label = r"y(x)")
plt.plot(solve1[0].t,np.exp(solve1[1]),"--", label = r"y_{eq}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(1,3,2)
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-10")
plt.xlabel("x")
plt.ylabel("y")
plt.yscale("log")
plt.plot(solve2[0].t,np.exp(solve2[0].y[0]),label = r"y(x)")
plt.plot(solve2[0].t,np.exp(solve2[1]),"--", label = r"y_{eq}")
plt.legend()

plt.subplot(1,3,3)
plt.xlabel("x")
plt.ylabel("y")
plt.yscale("log")
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-11")
plt.plot(solve3[0].t,np.exp(solve3[0].y[0]),label = r"y(x)")
plt.plot(solve3[0].t,np.exp(solve3[1]),"--", label = r"y_{eq}")
plt.legend()

"---------------------------------------------------------------"

plt.figure()
plt.subplot(1,3,1)
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-9")
plt.plot(solve1[0].t[:200],np.exp(solve1[0].y[0][:200]),label = r"y(x)")
plt.plot(solve1[0].t[:200],np.exp(solve1[1][:200]),"--", label = r"y_{eq}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(1,3,2)
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-10")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(solve2[0].t[:200],np.exp(solve2[0].y[0][:200]),label = r"y(x)")
plt.plot(solve2[0].t[:200],np.exp(solve2[1][:200]),"--", label = r"y_{eq}")
plt.legend()

plt.subplot(1,3,3)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"y and $y_{eq}$ plt for $\langle \sigma v \rangle$ = 1e-11")
plt.plot(solve3[0].t[:200],np.exp(solve3[0].y[0][:200]),label = r"y(x)")
plt.plot(solve3[0].t[:200],np.exp(solve3[1][:200]),"--", label = r"y_{eq}")
plt.legend()
plt.show()


def xf_finder(x,y):
    if len(x) != len(y):
        return 0
    else:
        for i, yy in enumerate(y):
            if yy <= 0.1*y[0]:
                return x[i]

def darkmatter(x,Y,cross):
    return 8.45e-11 * xf_finder(x,Y) /(np.sqrt(gstar) * cross)

darkmatter1 = darkmatter(solve1[0].t,np.exp(solve1[0].y[0]),cross1)
darkmatter2 = darkmatter(solve2[0].t,np.exp(solve2[0].y[0]),cross2)
darkmatter3 = darkmatter(solve3[0].t,np.exp(solve3[0].y[0]),cross3)

print(darkmatter1,darkmatter2,darkmatter3)

omegalimit = [0.12-0.05,0.12+0.05]
samplecross = np.linspace(1e-14, 1e-7,1000)


def crossfinder(omglim,x,cross):
    crosses = []
    kr = []
    dm = np.linspace(cross[0],cross[-1],len(cross))
    for k,j in enumerate(cross):
        solved = solve(x,j)
        dm[k] = darkmatter(solved[0].t,np.exp(solved[0].y[0]),j)
        if dm[k] >= omglim[0] and dm[k] <= omglim[-1]:
            crosses.append(j)
            kr.append(k)
        """
        if dm < omglim[0]:
            return crosses
            break
        """
    return crosses,dm,kr

section = crossfinder(omegalimit, [1,745], samplecross)
print(section[0])
print(section[0][0],section[0][-1])

plt.figure()
plt.title("Darkmatter as a function of cross section")
plt.xlabel(r"$\langle \sigma v \rangle$ [GeV^-2]")
plt.ylabel(r"$\Omega_{dm,0}h^2$")
plt.axhline(omegalimit[0], label = "Lower cross section limit", c = "r")
plt.axhline(omegalimit[-1], label = "Upper cross section limit", c = "r")
plt.plot(samplecross,section[1],label = "Darkmatter")
plt.ylim(omegalimit[0]-0.01,omegalimit[-1] + 0.01)
plt.xlim(2e-10,9e-10)
plt.legend()
plt.show()
