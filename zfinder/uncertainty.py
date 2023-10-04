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

def z_uncert(z, chi2, ssigma):
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
    
    Returns
    -------
    neg_uncert : float
        The left (- / negative) uncertainty

    pos_uncert : float
        The right (+ / positive) uncertainty
    """
    
    lowest_chi2 = argmin(chi2)
    best_z = z[lowest_chi2]
    min_chi2 = min(chi2)

    # Get points left and right of lowest_y that is above lowest_y + sigma
    left = lowest_chi2-1
    right = lowest_chi2+1

    # Create the axis arrays for the line between the left, lowest, and right point
    line_points_z = [z[left], best_z, z[right]]
    line_points_chi2 = [chi2[left], min_chi2, chi2[right]]
    
    # Find the parameters of the parabola that connects left, lowest, & right
    parabola_params, covars = curve_fit(lambda x, a: quadratic(x, a, b=best_z, c=min_chi2), line_points_z, line_points_chi2)

    # Calculate the z points above lowest_chi2 + sigma 
    neg_z, pos_z = solve_quadratic(min_chi2 + ssigma, *parabola_params, b=best_z, c=min_chi2)

    # Calculate the uncertainty in z
    neg_uncert_z = best_z - neg_z
    pos_uncert_z = pos_z - best_z
    return neg_uncert_z, pos_uncert_z