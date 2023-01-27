import numpy as np
from scipy.optimize import curve_fit

class uncertainty(object):
    def __init__(self, x, y, ssigma):
        """ 
        Method of calculating the uncertainty of a chi-squared vs x plot
        
        Parameters
        ----------
        x : list
            A list of x-axis values
        
        y : list
            A list of chi-squared values

        ssigma : float
            The squared significance level of the uncertainty. ssigma = sigma**2
        
        Returns
        -------
        unc_neg : float
            The '-' uncertainty
        
        unc_pos : float
            The '+' uncertainty
        """
        self.x = x
        self.y = y
        self.ssigma = ssigma

    @staticmethod
    def _quadratic(x, a, b, c):
        """ Fit a quadratic to 3 points closest to the minimum chi-squared """
        y = a*(x-b)**2 + c
        return y

    @staticmethod
    def _solve_quadratic(y, a, b, c):
        """ Solve the quadratic to find x at a certain sigma """
        neg = -np.sqrt((y-c)/a) + b
        pos = np.sqrt((y-c)/a) + b
        return neg, pos

    def calc_uncert(self):
        """ Caclulate the uncertainty of a chi-squared plot """
        
        lowest_y_index = np.argmin(self.y)
        min_x = self.x[lowest_y_index]
        min_y = min(self.y)

        # Get points left and right of lowest_y that is above lowest_y + sigma
        left = lowest_y_index-1
        right = lowest_y_index+1

        # Create the axis arrays for the line between the left, lowest, and right point
        points_x = [self.x[left], min_x, self.x[right]]
        points_y = [self.y[left], min_y, self.y[right]]
        
        # Find the parameters of the parabola that connects left, lowest, & right
        params, covars = curve_fit(lambda x, a: self._quadratic(x, a, b=min_x, c=min_y), points_x, points_y)

        # Calculate the x points above lowest_y + sigma 
        neg, pos = self._solve_quadratic(min_y + self.ssigma, *params, b=min_x, c=min_y)

        # Calculate the uncertainty in x
        unc_neg = min_x - neg
        unc_pos = pos - min_x
        return unc_neg, unc_pos