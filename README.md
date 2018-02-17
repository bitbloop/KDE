# Unsupervised Kernel Density Estimates with Mean-Shift Local Maxima Discovery

## Summary

Using unsupervised learning, the code finds local maxima of a Kernel Density Estimation function.

This Python code implementation first computes the Kernel Density Estimates for the normalized set of input data points, then it initializes candidate points for the local maxima, and using gradient ascent moves the candidate points along the gradient until they converge at local maxima.

Three different kernel functions are used to compute different Kernel Density Estimates: a Gaussian Kernel, an Uniform Kernel, and an Epanechnikov Kernel. In the Mean-Shift part of the code, only the Gaussian Kernel is used.

The filters require a filter bandwidth parameter to be specified. This bandwidth is computed using the Silverman's rule of thumb.

## Figure

![Mean-Shift](http://radosjovanovic.com/projects/git/kde.png)

Graph Legend:  
x axis - Value of Data Points  
y axis - Kernel Density Estimate  

Blue Points - Input Data  
Red Curve - Gaussian Kernel Density Estimation  
Red Line(s) - Approximate Local Maxima  
Black Curve - Uniform Kernel Density Estimation  
Green Curve - Epanechnikov Kernel Density Estimation  

## Authors

* **Rados Jovanovic** - *Initial work* - [bitbloop](https://github.com/bitbloop)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to everyone contributing to science!


