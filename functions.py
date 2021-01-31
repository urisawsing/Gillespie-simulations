 
 ##############################################
#			functions program with all			#
#	functions needed in the main algorithm		#
 ##############################################

#	importing the libraries for math (numpy) random variables (random) 
#	and for plotting purposes (matplotlib)
import numpy as np 
import random 
import matplotlib.pyplot as plt


def params():
	'''
	function returning the parameters needed for the simulation
	when called
	'''
	kd=10
	kdeg=2
	r=0.4
	E=5
	S=500
	a=3
	k1=(a*S)/(E*np.sqrt(kd))
	k2=kdeg
	R=(r*S)/(np.sqrt(kd))
	u11=(kdeg*S)/E
	b11=0.0008
	nu11=u11/(b11*S*S)
	return R,k1,k2,b11,u11,nu11,E,S



def Binom(n,p):
	'''
	simple function returning a sample of a binomial distribution
	with a determined n and p
	'''
	b=np.random.binomial(n,p)
	return b



def randomchoices():
	'''
	function returning two random float variables
	uniformly distributed between 0-1
	'''
	za=random.random()
	zb=random.random()
	return za,zb



def waiting_T(z,w):
	'''
	function used to calculate the waiting time
	it takes the z random sample and w weights and return
	the exponential distributed variable
	'''
	tau=(1/w)*np.log(1/z)
	return tau



def weights_sum(wv):
	'''
	function adding up all the weights from
	a weights vector, kind of adding all the values 
	from a vector in an scalar variable
	'''
	wsum=0
	for i in range(len(wv)):
		wsum=wsum+wv[i]
	return wsum



def weights_t(x1,x2):
	'''
	function calculating the weights according to the expressions
	on the pdf and returning an array with all of them
	'''
	R,k1,k2,b11,u11,nu11,E,S=params()   #calling the function params to obtain the constants
	w1=R+k1*x2
	w2=k2*x1
	w3=b11*x1*(x1-1)*(E-x2)
	w4=u11*x2
	return np.array([w1,w2,w3,w4])



def choice(we,e):
	'''
	function choosing the event happening by taking the weight just before the time t=tau*W_0, in this case this
	time t is e in the parameters and we the vector of weights
	'''
	wpre=0
	wnext=0
	success=0
	for i in range(len(we)):
		wnext=wnext+we[i]
		if(e<wnext and wpre<e):
			success=i
		wpre=wpre+we[i]
	return success



def rateG(j):
	'''
	function returning the updating for the Steady state gillespie, j is the "process" happening and 
	the function decides how will change X_1 with rate1 and X_2 with rate2
	'''
	rate1=0
	rate2=0
	if j==0:
		rate1=1
		rate2=0
	elif j==1:
		rate1=-1
		rate2=0
	elif j==2:
		rate1=-2
		rate2=1
	elif j==3:
		rate1=2
		rate2=-1
	return rate1,rate2


def updateG(j,x1,x2,t,tau):
	'''
	main function for the update of the variables, also for the steady state Gillespie,
	here is getting the rates from the function rateG and adding the result to the old variable value
	also is updating the time t with t+tau being tau the waiting time obtained by the random variable.
	j is the number of the process happening
	'''
	r1,r2=rateG(j)
	x1n=x1+r1
	x2n=x2+r2
	tn=t+tau
	return x1n,x2n,tn

def rateQ(j):
	'''
	Similarly to rateG but for the QSS approximation, here the only variable updated is the X_1 and so
	we only have rate1 calculated
	'''
	rate1=0
	if j==0:
		rate1=1
	elif j==1:
		rate1=-1
	elif j==2:
		rate1=-2
	elif j==3:
		rate1=2
	return rate1


def updateQ(j,x1,x2,t,tau):
	'''
	Also similar to updateG but for the QSS approximation, the main difference between this and updateG
	is that rate1 is obtained from the normal gillespie rateQ function but X_2 is updated through a binomial
	distribution sampling depending on the actual value of X_1
	'''
	R,k1,k2,b11,u11,nu11,E,S=params()
	tn=t+tau
	r1=rateQ(j)
	x1n=x1+r1
	n1=(x1/S)**2
	pi=(n1/(nu11+n1))
	x2n=Binom(E,pi)
	return x1n,x2n,tn



def plotting(x1v,x2v,tv):
	'''
	function for plotting purpouses mainly, it uses matplotlib to plot every simulation separately with both X_1 and X_2
	also computes the mean, standard deviation from numpy and finally returns the two boxplots of X_1 and X_2 for one simulation
	'''
	plt.plot(tv,x1v,label="x1") #plot of x1 time series
	plt.plot(tv,x2v,label="x2") #plot of x2 time series

	#formatting the plots properly
	plt.xlabel("time(s)")
	plt.ylabel("x value")
	plt.legend()
	plt.grid()
	plt.title("plot")
	plt.show()

	#some statistical values calculated, avg and std for each variable
	avgx=np.mean(x1v)
	dgx=np.std(x1v)
	avgy=np.mean(x2v)
	dy=np.std(x2v)
	print("Average for x_1",avgx)
	print("Average for x_2",avgy)

	#boxplots plotting in one same plot using matplotlib subplots and other formatting lines
	tot=[x1v,x2v]
	fig= plt.figure(1,figsize=(9,6))
	ax=fig.add_subplot(111)
	bp=ax.boxplot(tot) #plot of the two boxplots
	plt.show()



def plotboth(x1g,x2g,tg,x1q,x2q,tq,cg="red",cq="green"):
	'''
	function plotting the two simulations after being done, the xg variables are for the gillespie approximation
	and the xq for the quasi steady state approximation.
	The plots done here are the same time series as in function plotting() then a superposed plot with the four time
	series and finally the cumulative histogram for x1 variables in order to find similarities between distributions

	'''

	#plotting the time series for the normal Gillespie algorithm
	plt.plot(tg,x1g,label="x1")
	plt.plot(tg,x2g,label="x2")
	plt.title("SSA")
	plt.xlabel("time(s)")
	plt.ylabel("x value")
	plt.legend()
	plt.grid()
	plt.show()

	#plotting the time series for the QSS Gillespie algorithm
	plt.plot(tq,x1q,label="x1")
	plt.plot(tq,x2q,label="x2")
	plt.xlabel("time(s)")
	plt.ylabel("x value")
	plt.legend()
	plt.title("QSSA")
	plt.grid()
	plt.show()

	#plotting both QSS and normal Gillespie time series, differenced them by adding different colors and some transparency
	plt.plot(tg,x1g,label="SS1",color="red")
	plt.plot(tq,x1q,label="QSS1",color="orange",alpha=0.7)
	plt.plot(tg,x2g,label="SS2",color="green")
	plt.plot(tq,x2q,label="QSS2",color="blue",alpha=0.7)
	plt.grid()
	plt.legend()
	plt.show()

	#histogram plotting for both x1 variables, using a range of (0,60) and 60 bins being coherent with the discrete nature of the variables
	plt.hist(x1g,bins=60,range=(0,60),cumulative=True,histtype="step",color="red",label="Gillespie")
	plt.hist(x1q,bins=60,range=(0,60),cumulative=True,histtype="step",color="green",label="QSS approximation")
	plt.title("Cumulative histogram for X1")
	plt.legend()
	plt.xlabel("value of X1")
	plt.ylabel("frequency")
	plt.grid()
	plt.show()


def boxplotsboth(x1g,x2g,x1q,x2q):
	'''
	mixed boxplot plotting, in this function instead of putting X_1 and X_2 for one simulation we plotted the
	two same variables X_1  of the different algorithms in the same plot in order to compare if they are similar
	'''

	#calculating again some statistical values, average of gillespie alg in this case
	avgxg=np.mean(x1g)
	avgyg=np.mean(x2g)
	print("Average for x_1 in SS",avgxg)
	print("Average for x_2 in SS",avgyg)

	#here we calculated the avg of the QSS algorithm variables
	avgxq=np.mean(x1q)
	avgyq=np.mean(x2q)
	print("Average for x_1 in QSS",avgxq)
	print("Average for x_2 in QSS",avgyq)

	#here the standard deviation for Gillespie
	stdxg=np.std(x1g)
	stdyg=np.std(x2g)
	print("St. deviation for x_1 in SS",stdxg)
	print("St. deviation for x_2 in SS",stdyg)

	#and here for the QSS
	stdxq=np.std(x1q)
	stdyq=np.std(x2q)
	print("St. deviation for x_1 in QSS",stdxq)
	print("St. deviation for x_2 in QSS",stdyq)

	#defining the arrays of vectors for the boxplots
	totx1=[x1g,x1q]
	totx2=[x2g,x2q]

	#boxplots for variable X_1 
	fig= plt.figure(1,figsize=(9,6))
	ax=fig.add_subplot(111)
	ax.set_title("Boxplots for X1")
	bp=ax.boxplot(totx1)
	plt.show()

	#boxplots for variable X_2
	fig= plt.figure(1,figsize=(9,6))
	ax=fig.add_subplot(111)
	ax.set_title("Boxplots for X2")
	bp=ax.boxplot(totx2)
	plt.show()

