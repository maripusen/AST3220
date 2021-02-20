import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import *

class Inflation():
    def __init__(self):
        "intialise variables"
        self.tau = np.linspace(0,300,3000)
        self.psi = np.linspace(0,300,len(self.tau))
        self.dtau = (self.tau[1] - self.tau[0])
        self.psi = np.zeros(len(self.tau))
        self.huns = np.zeros(len(self.tau))
        self.dpsi = np.zeros(len(self.tau))

        self.psii = 3.1
        self.psi[0] = self.psii
        self.epsquared = (hbar*c**5)/(G)
        self.Hisquared = (8 *np.pi * G)/(3*c**2) * 0.5*((0.01**2 * self.epsquared)/(hbar * c)**3) * (self.psii**2 * self.epsquared)

    def potential(self,ppsi):
        V = 0.5 *((0.01**2 *self.epsquared)/(hbar*c)**3) * self.epsquared*ppsi**2
        return (hbar*c**3)/((self.Hisquared*self.epsquared)) * V

    def dpot(self,ppsi):
        return (0.01**2 *self.epsquared)/(hbar**2 *self.Hisquared)* ppsi

    def hsquared(self,ppsi,dpsi):
        return (8*np.pi)/3*(0.5*dpsi**2 + self.potential(ppsi))

    def slowroll(self):
        slow = self.psii - self.tau * (0.01 * np.sqrt(self.epsquared))/(hbar * np.sqrt(12 * np.pi) *np.sqrt(self.Hisquared))
        return slow

    def solve(self):
        self.huns[0] = 1 * self.dtau
        self.dpsi[0] = -1/3*(self.dpot(self.psii))
        for i in range(1,len(self.tau)):
            ddpsi = -3 * np.sqrt(self.hsquared(self.psi[i-1],self.dpsi[i-1])) * self.dpsi[i-1] - self.dpot(self.psi[i-1])
            self.dpsi[i] = self.dpsi[i-1] + ddpsi*self.dtau

            self.psi[i] = self.psi[i-1] + self.dpsi[i-1] * self.dtau

            self.huns[i] = self.huns[i-1] + np.sqrt(self.hsquared(self.psi[i],self.dpsi[i])) * self.dtau

        rate = (0.5 * (self.dpsi)**2 - self.potential(self.psi))/(0.5 * (self.dpsi)**2 + self.potential(self.psi))
        return self.psi, self.tau, self.huns, rate

cd = Inflation()

numeric = cd.solve()
slor = cd.slowroll()

plt.figure()
plt.xlabel(r"$\tau$",fontsize = 15)
plt.ylabel(r"$\psi$",fontsize = 15)
plt.plot(numeric[1],slor, label = "Slowroll")
plt.plot(numeric[1],numeric[0], label = "Numerical")
plt.legend(fontsize = 15)
plt.show()


plt.figure()
plt.xlabel(r"$\tau$",fontsize = 15)
plt.ylabel("e-foldings",fontsize = 15)
plt.plot(numeric[1],np.exp(numeric[2]), label = r"$\ln \left( \frac{a(\tau)}{a_i} \right)$")
plt.legend(fontsize = 15)
plt.show()


plt.figure()
plt.xlabel(r"$\tau$",fontsize = 15)
plt.plot(numeric[1], numeric[3], label = r"$\frac{\frac{1}{2} \left( \frac{d\psi}{d\tau}\right) - v}{\frac{1}{2} \left( \frac{d\psi}{d\tau}\right) + v}$")
plt.legend(fontsize = 15)
plt.show()
