import numpy as np
import matplotlib.pyplot as plt

# uniform kernel
def Ku(u):
	return (np.abs(u)<0.5).astype(np.float)

# gaussian kernel
def Ks(u):
	return np.exp(-0.5*np.square(u)) / np.sqrt(2*np.pi)

# epanechnikov kernel
def Ke(u):
	return (np.square(u)<=1).astype(np.float) * (1-np.square(u))             

# A mean-shift implementation
# x - data points
# k - the mean shift cluster initial points
# Kdx - partial derivative function to be used when taking a step
# h - filter bandwidth
# epsilon_error - loops until the squared shift is less than epsilon_error
def mean_shift(x, k, Kdx, h, epsilon_error):
	m=k.shape[0]			# number of points to perform the ascent with
	error=epsilon_error+1
	while (error >= epsilon_error):
		error=0
		# preform a gradient ascent
		for ki in range(0, m):
			dk=Kdx((k[ki]-x) / h)
			numerator=np.sum(dk * x)
			denominator=np.sum(dk)
			shift=(np.sum(np.divide(numerator,denominator)) - k[ki])
			k[ki]+=shift
			error += np.sum(np.abs(shift))
	return k

# entrypoint
def main(unused_argv):
	# generate input data 
	m=6 		# number of inputs
	x=np.random.uniform(low=0.0, high=1.0, size=(m))	# generate inputs

	# bandwidth
	h=np.std(x)*np.power((4/3/m),(1/5)); # Silverman's rule of thumb

	# compute the mean shift
	epsilon_error=0.000001 	# iteration stopping threshold
	k=x.copy() 				# make each local maximum candidate to start at data points
	mean_shifted_points_s = mean_shift(x, k, Ks, h, epsilon_error)
	

	# ----------------------------------
	# plot the density functions and the discovered local maximas
	
	# create points for plotting of density functions purposes
	i=np.arange(0,1,0.001).reshape(-1,1)		# denerate the datapoints for sampling the density functions
	kernels_uniform=(1/(m*h)) * np.sum( Ku( (i-x) / h), axis=1)
	kernels_gaussian=(1/(m*h)) * np.sum( Ks( (i-x) / h), axis=1)
	kernels_epanechnikov=(1/(m*h)) * np.sum( Ke( (i-x) / h), axis=1)
	# plot the data points
	plt.plot(x, np.zeros_like(x), "ro", color='blue')
	# plot the functions
	plt.plot(i, kernels_uniform, "b-", color='black')
	plt.plot(i, kernels_epanechnikov, "g-", color='green')
	plt.plot(i, kernels_gaussian, "r-", color='red')
	# plot the local maxima for the standard gaussian density function
	for ms in mean_shifted_points_s:
		plt.axvline(x=ms, color='red')
	
	plt.show()

	return 0

# main
if __name__ == "__main__":
	main(main)
