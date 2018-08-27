Tarr = []
N = 10000
n = 10
for i in range(100):
    U = LCG_rand2 = LCG(3, a, c, M, 10000)
    nobs,_ = np.histogram(U,bins=10)
    Tarr.append(((nobs-N/n)**2/(N/n)).sum())

N = 40
plt.plot([i for i in range(N)],[scipy.stats.chi2.pdf(i,df=n-1) for i in range(N)],label = "$\chi^2 $")
plt.plot([np.mean(Tarr), np.mean(Tarr)],[-1,1],'--g',label = "$T_{mean}$")
plt.plot([scipy.stats.chi2.ppf(0.975,df=9),scipy.stats.chi2.ppf(0.975,df=9)],[-1,1],'--r', label = "$95\%$ CI")
plt.plot([scipy.stats.chi2.ppf(0.025,df=9),scipy.stats.chi2.ppf(0.025,df=9)],[-1,1],'--r')
plt.ylim([0,0.11])
plt.xlim([0,25])
plt.ylabel("Propability")
plt.xlabel("Value")
plt.legend()
plt.show()
T = chi_square_distribution(LCG_rand, 10, usePlt=False)
print("Chi2 test: " + hypo_test_chi(T, 9, 0.975))
plt.close()