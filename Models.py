import numpy as np
import scipy as sp
import scipy.special as sc
from typing import Optional
from scipy.fft import fft, fftshift, ifftshift
import matplotlib.pyplot as plt

class BG:
    def __init__(self, bp, cp, bn, cn, N=4096, k = 5, Xmax=None):
        """
        Construct a bilateral‐Gamma PDF by inverting its characteristic function via FFT.

        Parameters
        ----------
        bp : float
            Positive‐jump scale (b₊)
        cp : float
            Positive‐jump shape (c₊)
        bn : float
            Negative‐jump scale (b₋)
        cn : float
            Negative‐jump shape (c₋)
        N : int, optional
            Number of FFT points (power of two). Default is 4096.
        
        After construction, `self.x` will be the grid on [-Xmax, Xmax),
        and `self.p` will be the corresponding normalized PDF values.
        """
        self.bp, self.cp, self.bn, self.cn = bp, cp, bn, cn
        self.N = N

        # 1) Estimate variance → σX, then set Xmax ≈ kσX
        if not Xmax:
            var_est  = cp * bp * bp + cn * bn * bn
            sigma_X  = np.sqrt(var_est)
            self.Xmax = k * sigma_X
        else:
            self.Xmax = Xmax    

        # 2) Compute T so that Δx = 2·Xmax / N  and  Δt = 2·T / N
        self.T = (np.pi * N) / (2.0 * self.Xmax)

        # 3) Build frequency grid t_k ∈ [−T, T), length N:
        dt = 2.0 * self.T / N
        t  = np.arange(-N//2, N//2) * dt  # centered at 0

        # 4) Evaluate φ_BG(t) on that grid:
        phi_vals = (1.0 - 1j * t * bp)**(-cp) * (1.0 + 1j * t * bn)**(-cn)

        # 5) Shift φ so that t=0 is at index 0 for the IFFT:
        phi_shifted = ifftshift(phi_vals)

        # 6) Perform IFFT to get an unnormalized f(x):
        #    IFFT{φ_shifted} * (Δt / 2π) * N
        raw_ifft     = fft(phi_shifted)
        unscaled_pdf = np.real(raw_ifft) * (dt / (2.0 * np.pi)) * N

        # 7) Shift back so x=0 is centered:
        pdf_vals = fftshift(unscaled_pdf)

        # 8) Build the spatial grid x ∈ [−Xmax, Xmax):
        dx = 2.0 * self.Xmax / N
        x  = np.linspace(-self.Xmax, self.Xmax - dx, N)

        # 9) Drop any tiny negative values and normalize:
        pdf_vals[pdf_vals < 0] = 0.0
        pdf_vals /= np.trapz(pdf_vals, x)

        # 10) Store final x and p:
        self.x = x
        self.p = pdf_vals

    def plot_pdf(self):
        """
        Plot the PDF using Matplotlib.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.x, self.p, color="tab:orange")
        plt.title("Bilateral Gamma PDF via FFT Inversion")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()


# class BG():

#     def __init__(self,
#                 k: int,
#                 M: list[float],
#                 params: list[float]
#     ):
#         self.k = k
#         self.M = M
#         self.y = np.linspace(M[0],M[1],2**k)
#         ip = (self.y > 0)
#         im = (self.y < 0)
#         yp = self.y[ip]
#         ym = -self.y[im]
#         self.delta = (M[1]-M[0])/(2**k-1)

#         self.bp = params[0]
#         self.cp = params[1]
#         self.bn = params[2]
#         self.cn = params[3]            
#         b = (1/self.bp+1/self.bn)
#         gammap = sc.gamma(self.cp)
#         gamman = sc.gamma(self.cn)
#         lam = 0.5*(self.cp+self.cn)
#         mu = 0.5*(self.cp+self.cn-1)
#         x = self.y*b
#         xp = x[ip]
#         xm = -x[im]
        
#         W = np.zeros(x.shape)
#         p = np.zeros(x.shape)
#         W[ip] = np.exp(-0.5*xp) * (xp**(mu+0.5)) * sc.hyperu(mu-lam+0.5, 1+2*mu, xp)
#         W[im] = np.exp(-0.5*xm) * (xm**(mu+0.5)) * sc.hyperu(mu-lam+0.5, 1+2*mu, xm)
#         p[ip] = ( (self.bp)**(-self.cp) ) * ( (self.bn)**(-self.cn) ) * ( (yp)**(0.5*(self.cp+self.cn)-1) ) * np.exp(-0.5*yp) * W[ip] / gammap
#         p[im] = ( (self.bp)**(-self.cp) ) * ( (self.bn)**(-self.cn) ) * ( (ym)**(0.5*(self.cp+self.cn)-1) ) * np.exp(-0.5*ym) * W[im] / gamman
#         self.p = p * self.delta / ( b**(0.5*(self.cp+self.cn)) )
#         self.p = p/sum(p)

#     def BGprice(self,
#                 S: list[float],
#                 k0: np.ndarray,
#                 k1: np.ndarray,
#                 r: float,
#                 T: float,
#                 alpha: float, lam: float, eta: float, N: int,
#     ) -> list[np.ndarray]:
        
#         bp = self.bp
#         bn = self.bn
#         cp = self.cp
#         cn = self.bn

#         beta = np.log(S[0])-lam*N/2
#         k = beta+(np.cumsum(np.ones(N,1))-1)*lam
#         u = (np.cumsum(np.ones(N,1))-1)*eta
#         w = np.ones(N,1)*eta
#         w[0] = w[0]/2
#         x = np.exp(-1j*beta*u)*self.Psi_BG(u,bp,cp,bn,cn,T,r,alpha,S[0])*w
#         Call = np.real((np.exp(-alpha*k)/np.pi)*sp.fft(x,N))
#         kk = np.log(k0)
#         C0 = np.interp(k,Call,kk)
#         P0 = C0 - S[0] + k0*np.exp(-r*T)

#         beta = np.log(S[1])-lam*N/2
#         k = beta+(np.cumsum(np.ones(N,1))-1)*lam
#         u = (np.cumsum(np.ones(N,1))-1)*eta
#         w = np.ones(N,1)*eta
#         w[0] = w[0]/2
#         x = np.exp(-1j*beta*u)*self.Psi_BG(u,bp,cp,bn,cn,T,r,alpha,S[0])*w
#         Call = np.real((np.exp(-alpha*k)/np.pi)*sp.fft(x,N))
#         kk = np.log(k1)
#         C1 = np.interp(k,Call,kk)
#         P1 = C1 - S[1] + k1*np.exp(-r*T)

#         return [C0, P0, C1, P1]

#     def Phi_BG(self,
#                 u: float,
#                 r: float,
#                 k: np.ndarray,
#                 T: float
#     ) -> list[float]:
#         bp = self.bptil
#         bn = self.bntil
#         cp = self.cptil
#         cn = self.bntil
#         d0 = 1j*u*(np.log(k)+T*(r+cp*np.log(1-bp)+cn*np.log(1+bn)))
#         Phi = np.exp(d0)*((1-1j*u*bp)^(-T*cp))*((1+1j*u*bn)^(-T*cn))
#         return Phi

#     def Psi_BG(self,
#                u: float,
#                r: float,
#                k: np.ndarray,
#                T: float,
#                alpha: float
#         ) -> list[float]: 
#         bp = self.bptil
#         bn = self.bntil
#         cp = self.cptil
#         cn = self.bntil
#         Psi = (np.exp(-r*T)/((alpha+1j*u)/(alpha+1j*u+1)))/self.Phi_BG(u-(alpha+1)*1j,bp,cp,bn,cn,T,k,r)
#         return Psi