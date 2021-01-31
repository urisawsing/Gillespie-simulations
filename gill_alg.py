 #########################################
#					  #
#            gillespie algorithm 	  #
#  	for a self-gene regulating system #
#					  #
#		main program		  #
 #########################################


#Developed by Oriol Fernández 31/01/2021

'''
	importing the libraries for the main program
	math library (numpy) 
	random variables library (random) 
	plotting library (matplotlib)
	functions is the modular code with all the functions needed in the main code
	scipy used for the kolmogorov smirnov test
	time library for performance purpouses
'''
import numpy as np 
import random
import matplotlib.pyplot as plt
import functions as f
from scipy import stats
import time




def Gillespie(mi,tmax=100000):
	'''
	main function for the Steady state Gillespie algorithm, as args we have mi being the max iterations allowed
	as first termination condition and for kwargs the tmax setted as 1e5 by default.
	'''

	#initializing the variable vectors for x1,x2 and time
	x1vec=[]
	x2vec=[]
	tvec=[]

	#defining the initial conditions and some auxiliar variables like xnew for the "next step" position
	# or it as the iterative variable
	maxiter=mi
	x1=0.0
	x2=0.0
	t=0.0
	x1new=0.0
	x2new=0.0
	it=0

	#appending the initial values in the variable vectors
	x1vec.append(x1new)
	x2vec.append(x2new)
	tvec.append(t)

	print("Entering the Gillespie main loop")
	#main loop of the gillespie algorithm, only exiting when surpassing the maxiter or the tmax
	while(t<tmax and it<maxiter ):


		wvec=f.weights_t(x1,x2)						#calculating the vector of weights depending on the value of x1,x2

		W_0=f.weights_sum(wvec)						#calculating the sum of weights W0 from the weights vector

		t1,t2=f.randomchoices()						#sampling two times out of the uniformly distributed 0-1 random variable

		T=f.waiting_T(t1,W_0)						#calculating the waiting time from the first time sample and the W_0

		event=t2*W_0							#using the second waiting time to get which reaction will happen

		choosing=f.choice(wvec,event)					#choosing this reaction using the previous "event" variable and the vector of weights

		x1new,x2new,t=f.updateG(choosing,x1,x2,t,T)			#updating the value of the variables x1,x2,t

		#print("updated position is",x1new,x2new,t) 			#uncomment for getting the x value on every step


		#appending the new variable values in the vectors
		x1vec.append(x1new)
		x2vec.append(x2new)
		tvec.append(t)

		#redefining the x variables out of the xnew and updating the iteration variable
		x1=x1new
		x2=x2new
		it=it+1

	#f.plotting(x1vec,x2vec,tvec)						#uncomment for getting the plots at the end of every simulation
	print("steady state gillespie finished")
	return x1vec,x2vec,tvec							#returning all the vectors


	

def QGillespie(mi,tmax=100000,seed=42):
	'''
	main function for the Quasi Steady state Gillespie algorithm, as args we have mi being the max iterations allowed
	as first termination condition and for kwargs the tmax setted as 1e5 by default.
	'''

	#initializing the variable vectors for x1,x2 and time

	x1vec=[]
	x2vec=[]
	tvec=[]

	#defining the initial conditions and some auxiliar variables like xnew for the "next step" position
	# or it as the iterative variable
	maxiter=mi
	x1=0.0
	x2=0.0
	t=0.0
	x1new=0.0
	x2new=0.0
	it=0


	#appending the initial values in the variable vectors
	x1vec.append(x1new)
	x2vec.append(x2new)
	tvec.append(t)


	print("Entering QSS Gillespie algorithm")
	#main loop of the gillespie algorithm, only exiting when surpassing the maxiter or the tmax
	while(t<tmax and it<maxiter ):

		wvec=f.weights_t(x1,x2)						#calculating the vector of weights depending on the value of x1,x2

		W_0=f.weights_sum(wvec)						#calculating the sum of weights W0 from the weights vector

		t1,t2=f.randomchoices()						#sampling two times out of the uniformly distributed 0-1 random variable

		T=f.waiting_T(t1,W_0)						#calculating the waiting time from the first time sample and the W_0

		event=t2*W_0							#using the second waiting time to get which reaction will happen

		choosing=f.choice(wvec,event)					#choosing this reaction using the previous "event" variable and the vector of weights



		x1new,x2new,t=f.updateQ(choosing,x1,x2,t,T)			#updating the value of the variables x1,x2,t using the QSS criterion

		#print("updated position is",x1new,x2new,t) 			#uncomment for getting the x value on every step


		#appending the new variable values in the vectors
		x1vec.append(x1new)
		x2vec.append(x2new)
		tvec.append(t)

		#redefining the x variables out of the xnew and updating the iteration variable
		x1=x1new
		x2=x2new
		it=it+1

	#f.plotting(x1vec,x2vec,tvec)						#uncomment for getting the plots at the end of every simulation
	print(" quasi steady state gillespie finished")
	return x1vec,x2vec,tvec							#returning all the vectors



#main loop of the program	

start=time.time()								#initializing the time for performance purpouses


x1g,x2g,tg=Gillespie(500000)							#function calling the Gillespie algorithm maxiter=500000
gillespiet=time.time()								#saving the time elapsed for the Gillespie algorithm

x1q,x2q,tq=QGillespie(500000)							#function calling the QSS Gillespie algorithm maxiter=500000
end=time.time()									#saving the time elapsed for the QSS Gillespie algorithm




#calculating and printing the elapsed time for every algorithm
gtime=gillespiet-start
qtime=end-gillespiet
print("elapsed  G time",gtime)
print("elapsed  Q time",qtime)



#calling plotting functions and boxplotting functions with the vectors returned by the algorithms
f.plotboth(x1g,x2g,tg,x1q,x2q,tq)
f.boxplotsboth(x1g,x2g,x1q,x2q)




'''
testing the samples with the kolmogorov-smirnov test
the KS is the statistic and pv the Pvalue, after this 
printing out the results
'''
KS,pv=stats.ks_2samp(x1g,x1q)
KS2,pv2=stats.ks_2samp(x2g,x2q)
print("Kolmogorov-Smirnov test for x1 variable, statistic:",KS,"p-value",pv)
print("Kolmogorov-Smirnov test for x2 variable, statistic:",KS2,"p-value",pv2)




