from numpy import sqrt, argmin
from scipy.optimize import curve_fit

def quadratic(x, a, b, c):
    """ Fit a quadratic to 3 points closest to the minimum chi-squared """
    y = a*(x-b)**2 + c
    return y

def solve_quadratic(y, a, b, c):
    """ Solve the quadratic to find x at a certain sigma """
    neg = -sqrt((y-c)/a) + b
    pos = sqrt((y-c)/a) + b
    return neg, pos

def z_uncert(z, chi2, ssigma, reduction=False):
    """
    Caclulate the uncertainty in the best fitting redshift

    Parameters
    ----------
    z : list
        A list of redshift values
    
    chi2 : list
        A list of chi-squared values

    ssigma : float
        The squared significance level of the uncertainty. ssigma = sigma**2
    
    reduction : bool, optional
        Choose to reduce all chi2 values such that the minimum chi2 is 0. Default = False
    
    Returns
    -------
    neg_uncert : float
        The left (- / negative) uncertainty

    pos_uncert : float
        The right (+ / positive) uncertainty
    """
    
    lowest_y_index = argmin(chi2)
    min_x = z[lowest_y_index]
    min_y = min(chi2)
    
    if reduction:
        chi2 = [i - min_y for i in chi2]

    # Get points left and right of lowest_y that is above lowest_y + sigma
    left = lowest_y_index-1
    right = lowest_y_index+1

    # Create the axis arrays for the line between the left, lowest, and right point
    points_x = [z[left], min_x, z[right]]
    points_y = [chi2[left], min_y, chi2[right]]
    
    # Find the parameters of the parabola that connects left, lowest, & right
    params, covars = curve_fit(lambda x, a: quadratic(x, a, b=min_x, c=min_y), points_x, points_y)

    # Calculate the x points above lowest_y + sigma 
    neg, pos = solve_quadratic(min_y + ssigma, *params, b=min_x, c=min_y)

    # Calculate the uncertainty in x
    neg_uncert = min_x - neg
    pos_uncert = pos - min_x
    return neg_uncert, pos_uncert