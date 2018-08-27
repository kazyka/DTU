import numpy as np
from matplotlib import pyplot as plt
import math


class BlockingSystem():
    def __init__(self, n, serviceTimes):
        self.n = n
        self.availUnits = [True] * n
        self.queueTime = [0] * n
        self.rejectedCustomers = 0
        self.serviceTimes = serviceTimes
        self.customerCount = 0

    def updateQueues(self, time):
        for i in range(self.n):
            if not (self.availUnits[i]):
                if time > self.queueTime[i]:
                    self.availUnits[i] = True

    def customerArrival(self, time):
        self.updateQueues(time)
        foundQueue = False
        qCount = 0
        while qCount < self.n and not (foundQueue):
            if self.availUnits[qCount]:
                self.availUnits[qCount] = False
                self.queueTime[qCount] = time + self.serviceTimes[self.customerCount]
                foundQueue = True
            qCount += 1

        if not (foundQueue):
            self.rejectedCustomers += 1
        self.customerCount += 1
        return foundQueue

    def getNrRejected(self):
        return self.rejectedCustomers


def print_statistics(mean, lower, upper):
    for i in range(len(mean)):
        print('{0:.0f} & {1:.4f} & {2:.4f} & {3:.4f}'.format(i, lower[i] / 100, mean[i] / 100, upper[i] / 100))


if __name__ == '__main__':
    simNR = 10000
    nrQueues = 10
    nrTrials = 10

    ## exponential
    # Mean service time = 8 => beta = 8 in exponential
    # mean arrival time = 1 => lambda = 1 in poisson
    means = []
    upperConf = []
    lowerConf = []
    k = 1
    while k <= nrQueues:
        nrBlocked = []
        for t in range(nrTrials):
            current_time = 0
            arrivalProcess = np.random.exponential(1, simNR)
            serviceTimes = np.random.exponential(8, simNR)
            myBS = BlockingSystem(k, serviceTimes)
            for i in range(simNR):
                current_time += arrivalProcess[i]
                myBS.customerArrival(current_time)
            nrBlocked.append(myBS.getNrRejected())
        meanBlocked = np.mean(nrBlocked)
        varBlocked = np.var(nrBlocked)
        means.append(meanBlocked)
        upperConf.append(meanBlocked + np.sqrt(varBlocked) / np.sqrt(nrTrials) * 1.96)
        lowerConf.append(meanBlocked - np.sqrt(varBlocked) / np.sqrt(nrTrials) * 1.96)
        k += 1
    means = np.asarray(means)
    upperConf = np.asarray(upperConf)
    lowerConf = np.asarray(lowerConf)
    t = np.linspace(1, nrQueues, nrQueues)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, means / simNR * 100)
    ax.plot(t, upperConf / simNR * 100, '--', c='r')
    ax.plot(t, lowerConf / simNR * 100, '--', c='r')
    plt.show()
    print('Exponential-Poisson')
    print_statistics(means, lowerConf, upperConf)

    ##calculating theoretical values
    arriv_int = 1
    service_time = 8
    A = arriv_int * service_time
    n = 10
    denominator = 0
    for i in range(n + 1):
        denominator += A ** i / (math.factorial(i))
    print(denominator)
    B = (A ** n / math.factorial(n)) / denominator
    print('B: ' + str(B))

    ### erlang
    ##choosing k=1 and theta=1 to get a mean of 1
    # means = []
    # upperConf = []
    # lowerConf = []
    # k=1
    # while k<=nrQueues:
    #    nrBlocked = []
    #    for t in range(nrTrials):
    #        current_time = 0
    #        arrivalProcess = np.random.gamma(1,1,simNR)
    #        serviceTimes = np.random.exponential(1,simNR)
    #        myBS = BlockingSystem(k,serviceTimes)
    #        for i in range(simNR):
    #            current_time += arrivalProcess[i]
    #            myBS.customerArrival(current_time)
    #        nrBlocked.append(myBS.getNrRejected())
    #    meanBlocked = np.mean(nrBlocked)
    #    varBlocked = np.var(nrBlocked)
    #    means.append(meanBlocked)
    #    upperConf.append(meanBlocked+np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    lowerConf.append(meanBlocked-np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    k+=1

    # means = np.asarray(means)
    # upperConf = np.asarray(upperConf)
    # lowerConf = np.asarray(lowerConf)
    # t = np.linspace(1,nrQueues,nrQueues)
    ##fig,ax = plt.subplots(1,1)
    ##ax.plot(t,means/simNR*100)
    ##ax.plot(t,upperConf/simNR*100,'--',c='r')
    ##ax.plot(t,lowerConf/simNR*100,'--',c='r')
    ##plt.show()

    ## hyper exponential
    # choosing k=1 and theta=1 to get a mean of 1
    # p = [0.8,0.2]
    # lambd = [0.8333,5]
    # means = []
    # upperConf = []
    # lowerConf = []
    # k=1
    # while k<=nrQueues:
    #    nrBlocked = []
    #    for t in range(nrTrials):
    #        current_time = 0
    #        arrivalProcess1 = np.random.exponential(lambd[0],int(p[0]*simNR))
    #        arrivalProcess2 = np.random.exponential(lambd[1],int(p[1]*simNR))
    #        arrivalProcess = np.append(arrivalProcess1,arrivalProcess2)
    #        np.random.shuffle(arrivalProcess)
    #        serviceTimes = np.random.exponential(1,simNR)
    #        myBS = BlockingSystem(k,serviceTimes)
    #        for i in range(simNR):
    #            current_time += arrivalProcess[i]
    #            myBS.customerArrival(current_time)
    #        nrBlocked.append(myBS.getNrRejected())
    #    meanBlocked = np.mean(nrBlocked)
    #    varBlocked = np.var(nrBlocked)
    #    means.append(meanBlocked)
    #    upperConf.append(meanBlocked+np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    lowerConf.append(meanBlocked-np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    k+=1

    # means = np.asarray(means)
    # upperConf = np.asarray(upperConf)
    # lowerConf = np.asarray(lowerConf)
    # t = np.linspace(1,nrQueues,nrQueues)
    # fig,ax = plt.subplots(1,1)
    # ax.plot(t,means/simNR*100)
    # ax.plot(t,upperConf/simNR*100,'--',c='r')
    # ax.plot(t,lowerConf/simNR*100,'--',c='r')
    # plt.show()

    ## constant service time

    # serviceTime = 20
    # means = []
    # upperConf = []
    # lowerConf = []
    # k=1
    # while k<=nrQueues:
    #    nrBlocked = []
    #    for t in range(nrTrials):
    #        current_time = 0
    #        arrivalProcess =  np.random.poisson(1,simNR)
    #        serviceTimes = np.ones((simNR,1))*serviceTime
    #        myBS = BlockingSystem(k,serviceTimes)
    #        for i in range(simNR):
    #            current_time += arrivalProcess[i]
    #            myBS.customerArrival(current_time)
    #        nrBlocked.append(myBS.getNrRejected())
    #    meanBlocked = np.mean(nrBlocked)
    #    varBlocked = np.var(nrBlocked)
    #    means.append(meanBlocked)
    #    upperConf.append(meanBlocked+np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    lowerConf.append(meanBlocked-np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    k+=1

    # means = np.asarray(means)
    # upperConf = np.asarray(upperConf)
    # lowerConf = np.asarray(lowerConf)
    # t = np.linspace(1,nrQueues,nrQueues)
    # fig,ax = plt.subplots(1,1)
    # ax.plot(t,means/simNR*100)
    # ax.plot(t,upperConf/simNR*100,'--',c='r')
    # ax.plot(t,lowerConf/simNR*100,'--',c='r')
    # plt.show()

    ## Pareto service time

    # serviceTime = 20
    # means = []
    # upperConf = []
    # lowerConf = []
    # k=1
    # while k<=nrQueues:
    #    nrBlocked = []
    #    for t in range(nrTrials):
    #        current_time = 0
    #        arrivalProcess =  np.random.poisson(1,simNR)
    #        serviceTimes = np.random.pareto(2.05,simNR)
    #        myBS = BlockingSystem(k,serviceTimes)
    #        for i in range(simNR):
    #            current_time += arrivalProcess[i]
    #            myBS.customerArrival(current_time)
    #        nrBlocked.append(myBS.getNrRejected())
    #    meanBlocked = np.mean(nrBlocked)
    #    varBlocked = np.var(nrBlocked)
    #    means.append(meanBlocked)
    #    upperConf.append(meanBlocked+np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    lowerConf.append(meanBlocked-np.sqrt(varBlocked)/np.sqrt(nrTrials)*1.96)
    #    k+=1

    # means = np.asarray(means)
    # upperConf = np.asarray(upperConf)
    # lowerConf = np.asarray(lowerConf)
    # t = np.linspace(1,nrQueues,nrQueues)
    # fig,ax = plt.subplots(1,1)
    # ax.plot(t,means/simNR*100)
    # ax.plot(t,upperConf/simNR*100,'--',c='r')
    # ax.plot(t,lowerConf/simNR*100,'--',c='r')
    # plt.show()


