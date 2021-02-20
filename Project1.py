import numpy as np
import matplotlib.pyplot as plt


c = 299792.458
h = 0.7
H0 = 100 * h
Gpc = 3.26163344 * 10**9


def reading(file):
    data = open(file ,'r')
    lines = data.readlines()
    n = np.int((np.shape(lines)[0]))
    zdata = np.zeros(n); dLdata= np.zeros(n)
    errordata = np.zeros(n)
    for i, line in enumerate(lines):
        zdata[i], dLdata[i], errordata[i] = line.split()

    return zdata, dLdata,errordata

def omgs(omglam,omgmo):
    omgk0 = 1- (omglam + omgmo)
    return omgk0

def friedmann(z, omgmo, omgr0, omgk0,omglam):
    f = H0 * np.sqrt(omgmo * (1+z)**3 + omglam + omgr0 * (1+z)**4 + omgk0 * (1+z)**2)
    return f

def friedmann2(z,omgmo,omgrc,omgk0):
    f = H0*np.sqrt((np.sqrt(omgmo*(1+z)**3+ omgrc)+np.sqrt(omgrc))**2+omgk0*(1+z)**2)
    return f

def omgs2(omgrc,omgmo):
    print(np.shape(omgmo))
    return 1-(np.sqrt(omgmo+ omgrc)+np.sqrt(omgrc))**2

def hint(z, omgmo, omgr0, omgk0,omglam):
    dz = 1/1000
    zmrk = np.linspace(0,z,int(z/dz))
    hit = np.sum(H0/ friedmann(zmrk,omgmo,omgr0,omgk0, omglam)) * dz
    return  hit

def hint2(z, omgmo, omgrc, omgk0):
    dz = 1/1000
    zmrk = np.linspace(0,z,int(z/dz))
    hit = np.sum(H0/ friedmann2(zmrk,omgmo,omgrc,omgk0)) * dz
    return  hit

def Sk(omgk0,z,omgr0,omgmo, omglam):
    if isinstance(z,float):
        if omgk0 < 0:
            # k > 0
            sk = np.sin(np.sqrt(abs(omgk0))* hint(z,omgmo,omgr0,omgk0,omglam))
        elif omgk0 > 0:
            # k < 0
            sk = np.sinh(np.sqrt(abs(omgk0))* hint(z,omgmo,omgr0,omgk0,omglam))
        else:
            sk = hint(z,omgmo,omgr0,omgk0,omglam)

    else :
        sk = 0
        for i in range(0, len(z)):
            if omgk0 < 0:
        # k > 0
                sk = np.sin(np.sqrt(abs(omgk0))* hint(z[i],omgmo,omgr0,omgk0,omglam))
            elif omgk0 > 0:
        # k < 0
                sk = np.sinh(np.sqrt(abs(omgk0))* hint(z[i],omgmo,omgr0,omgk0,omglam))
            else:
                sk = hint(z[i],omgmo,omgr0,omgk0,omglam)
    return sk

def Sk2(omgk0,z,omgrc,omgmo):
    if isinstance(z,float):
        if omgk0 < 0:
            # k > 0
            sk = np.sin(np.sqrt(abs(omgk0))* hint2(z,omgmo,omgrc,omgk0))
        elif omgk0 > 0:
            # k < 0
            sk = np.sinh(np.sqrt(abs(omgk0))* hint2(z,omgmo,omgrc,omgk0))
        else:
            sk = hint2(z,omgmo,omgrc,omgk0)

    else :
        sk = 0
        for i in range(0, len(z)):
            if omgk0 < 0:
        # k > 0
                sk = np.sin(np.sqrt(abs(omgk0))* hint2(z[i],omgmo,omgrc,omgk0))
            elif omgk0 > 0:
        # k < 0
                sk = np.sinh(np.sqrt(abs(omgk0))* hint2(z[i],omgmo,omgrc,omgk0))
            else:
                sk = hint2(z[i],omgmo,omgrc,omgk0)
    return sk

def lumdis(z, omglam, omgmo, omgr0):
    omgk0 = omgs(omglam, omgmo)
    assert np.all(friedmann(z,omgmo, omgr0, omgk0, omglam)) > 0, "Friedmann equation is not positive"
    if omgk0 == 0:
        dl = c*(1+z)/(H0) * Sk(omgk0,z,omgr0,omgmo,omglam)
    else:
        dl = c*(1+z)/(H0 * np.sqrt(abs(omgk0))) * Sk(omgk0,z,omgr0,omgmo,omglam)
    return dl

def lumdis2(z, omgmo, omgrc):
    omgk0 = omgs2(omgrc, omgmo)
    assert np.all(friedmann2(z,omgmo, omgrc, omgk0)) > 0, "Friedmann equation is not positive"
    if omgk0 == 0:
        dl = c*(1+z)/(H0) * Sk2(omgk0,z,omgr0,omgmo,omglam)
    else:
        dl = c*(1+z)/(H0 * np.sqrt(abs(omgk0))) * Sk2(omgk0,z,omgrc,omgmo)
    return dl

def likelihood(omgmo,omglam,z,omgr0,dldata,error):
    return sum((((lumdis(z,omglam,omgmo,omgr0)/1000)-dldata)**2)/errordata**2)

def likelihood2(omgmo,z,omgrc,dldata,error):
    return sum((((lumdis2(z,omgmo,omgrc)/1000)-dldata)**2)/errordata**2)

def chisquare(dlp,dla,sigma,omgr0):
    inp1 = np.linspace(0.2,1,N)
    inp2 = np.linspace(-0.14,1,N)
    chi = np.zeros((len(inp1),len(inp2)))
    for i in range(0,len(inp1)):
        #print(i)
        for j in range(0,len(inp2)):
            chi[i,j] = likelihood(inp1[i],inp2[j],dlp,omgr0,dla,sigma)
    return chi

def chisquare2(dlp,dla,sigma,omgrc):
    inp1 = np.linspace(0.2,1,N)
    inp2 = np.linspace(-0.14,1,N)
    chi = np.zeros((len(inp1),len(inp2)))
    for i in range(0,len(inp1)):
        #print(i)
        for j in range(0,len(inp2)):
            chi[i,j] = likelihood2(inp1[i],dlp,inp2[j],dla,sigma)
    return chi


z_data, dldata, errordata = reading("sndata.txt")

N = 31
redshift = np.linspace(0,2,N)
omglam = np.linspace(0.2,1,N)
omgmo = np.linspace(-0.14,1,N)
omgrc = np.linspace(-0.14,1,N)

#print(likelihood(omgmo[9],omglam[3],z_data,np.zeros(N),dldata,errordata))


chi = chisquare(z_data,dldata,errordata,0)
meshx,meshy = np.meshgrid(omgmo,omglam)
chi -= np.amin(chi)
#print(chi)
plt.contour(meshx,meshy,chi)#, levels = [2.3,6.17,11.8], label = "Contour plot for the LambdaCDM model likelihood")
plt.legend()
plt.show()

"""
chi2 = chisquare2(z_data,dldata,errordata,0)
meshx,meshy = np.meshgrid(omgmo,omgrc)
chi2 -= np.amin(chi2)

plt.contour(meshx,meshy,chi2, levels = [2.3,6.17,11.8], label = "Contour plot for the DGP model likelihood")
plt.legend()
plt.show()
"""

#plt.plot(z_data, lumdis(z_data,0.59,0.26,0)/c * H0,  label = "Simulated LambdaCDM")
#plt.plot(z_data,lumdis2(z_data,0.59,0.26)/c*H0,label = "Simulated DGP")
#plt.legend()
#plt.show()
