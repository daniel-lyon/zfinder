import csv
import numpy as np
import matplotlib.pyplot as plt

# snrs = [[7.07, 6.46, 3.49], [4.3], [6.55, 6.13], [4.64, 3.89, 3.44], [7.02, 6.45, 3.5]]
# pixels = [[5, 4, 8], [4], [5, 4], [], [5, 4, 8]]
# peaks = [[308, 36, 283], [305], [308, 36], [383, 368, 299], [308, 36, 284]]

snrs = [[7.07, 6.46, 3.49], [6.51, 5.48], [3.48], [5.82, 4.58], [4.16, 3.13, 3.12], [3.08], [6.05, 5.45, 4.36], [5.04, 4.2], [3.31, 3.19, 3.03, 4.18], [4.78, 3.25], [5.49, 4.26, 4.19, 3.05], [3.87, 3.6], [3.68], [3.74, 3.44, 3.17, 3.34], [3.06], [5.66, 5.19, 3.62, 3.1], [3.49, 3.18], [3.38, 3.74], [4.96, 3.05], [5.26, 3.27], [3.44], [4.41, 3.64, 3.63, 3.39], [6.81, 6.44, 3.8], [5.17, 3.95, 3.6], [3.66, 3.11], [3.26, 3.71], [6.88], [3.59], [3.73, 3.54, 3.49, 3.31, 3.15], [5.32, 3.4, 3.11], [4.32, 3.4, 3.28], [5.4, 4.44, 4.04, 3.63, 3.05, 3.11], [3.19], [3.78, 3.48, 3.11, 3.27], [4.72, 4.3, 3.96], [5.61, 3.93, 3.85, 3.77, 3.01], [3.35, 3.03], [5.1, 4.04], [5.16, 3.67, 3.78, 3.45, 3.57], [4.53, 6.07, 4.7], [7.86, 3.39, 3.14], [6.38, 5.43], [3.31, 3.13], [], [4.5, 3.18, 3.13], [4.36, 3.84], [4.83, 4.12, 3.24], [4.16, 3.47], [3.36, 
3.16, 3.08], [3.83, 3.27, 3.02, 3.4, 3.13], [3.96, 3.89], [4.38, 3.99, 3.42], [4.67, 3.91], [3.19, 3.04], [3.12], [3.5, 3.1, 3.06], [3.65], [3.25], [3.98, 3.94, 3.45, 3.27], [4.46], [6.96, 5.54, 3.69], [4.74, 3.89, 5.1, 3.44, 3.44, 3.09], [5.43, 6.06, 4.47, 3.57], [3.34], [3.62, 3.22, 3.28, 3.06], [3.38, 3.37, 3.39, 3.11], [6.25, 5.65, 3.22], [], [3.42, 3.4], [3.59, 3.01], [4.58, 3.07], [4.44, 4.36, 4.22, 3.26], 
[5.53, 4.17], [], [5.61, 3.73, 3.81, 3.52, 3.2], [3.01], [4.21, 3.74, 3.32], [4.01], [4.15, 3.23, 3.1], [6.52, 6.24, 3.31], [3.53], [], [3.91, 3.38], [4.53, 3.07], [3.31, 3.0], [4.62, 3.69, 3.06, 3.11], [4.79, 4.25, 3.21], [5.8], [4.74, 4.08], [3.78], [6.21, 3.28], [3.77, 3.9], [5.61, 4.45], [3.75, 3.38], [3.6, 3.25, 3.01], [5.31, 4.58, 3.37], [4.3], [5.31, 4.12], [], [4.58], [4.36, 4.92, 4.58, 3.16, 3.2]]

pixels = [[5, 4, 8], [4, 3], [4], [4, 3], [1, 3, 1], [5], [4, 6, 9], [4, 3], [5, 3, 2, 3], [4, 2], [6, 7, 2, 6], [3, 1], [], [6, 5, 2, 1], [4], [4, 6, 9, 6], [3, 8], [4, 2], [8, 1], [3, 2], [6], [6, 7, 5, 1], [6, 4, 8], [1, 3, 2], [3, 2], [5, 3], [4], [6], [5, 3, 2, 2, 3], [1, 4, 3], [2, 4, 6], [], [8], [5, 3, 4, 2], [6, 5, 3], [8, 3, 5, 4, 5], [4, 5], [5, 3], [], [4, 6, 9], [4, 7, 4], [5, 3], [6, 4], [], [4, 6, 5], [4, 4], [5, 4, 5], [5, 3], [3, 2, 3], [5, 2, 2, 0, 1], [6, 4], [3, 5, 2], [5, 5], [11, 7], [4], [4, 5, 3], [4], [2], [5, 3, 5, 7], [6], [6, 4, 2], [9, 4, 6], [4, 6, 9, 6], [3], [5, 3, 3, 4], [4, 5, 2, 4], [5, 4, 0], [], [5, 6], [6, 5], [7, 3], [7, 4, 9, 2], [4, 3], [], [8, 5, 7, 4, 2], [7], [], [3], [1, 1, 4], [5, 5, 8], [0], [], [4, 7], [9, 8], [6, 3], [4, 5, 2, 1], [4, 4, 5], [6], [3, 4], [5], [3, 0], [5, 2], [5, 4], [3, 4], [5, 7, 5], [8, 7, 3], [5], [4, 4], [], [2], [8, 7, 2, 5, 2]]

peaks = [[308, 36, 283], [36, 307], [305], [36, 307], [378, 34, 328], [355], [36, 308, 283], [36, 307], [291, 379, 358, 263], [305, 323], [373, 314, 244, 279], [271, 379], [381], [324, 95, 351, 131], [321], [36, 308, 284, 365], [375, 75], [328, 121], [96, 334], [315, 20], [331], [325, 128, 368, 75], [308, 36, 284], [363, 382, 289], [123, 96], [358, 382], [326], [339], [355, 113, 382, 260, 333], [244, 281, 313], [276, 299, 258], [382, 333, 261, 112, 356, 283], [140], [249, 306, 274, 70], [362, 139, 265], [312, 330, 346, 362, 271], [324, 375], [36, 306], [381, 355, 261, 113, 283], [36, 311, 283], [326, 248, 269], [36, 307], [141, 336], [], [315, 20, 349], [244, 281], [35, 307, 18], [268, 76], [247, 134, 361], [98, 236, 313, 257, 288], [139, 55], [264, 139, 304], [276, 297], [43, 349], [147], [277, 351, 253], [85], [368], [368, 74, 14, 127], [60], [308, 36, 281], [283, 36, 311, 383, 261, 319], [36, 310, 283, 364], [92], [74, 365, 266, 340], [263, 346, 303, 361], [308, 36, 280], [], [276, 36], [139, 266], [335, 275], [34, 338, 320, 365], [36, 306], [], [312, 346, 362, 330, 49], [365], [382, 283, 261], [377], [270, 363, 332], [308, 36, 284], [72], [], [337, 95], [277, 36], [332, 67], [325, 268, 348, 366], [36, 307, 18], [380], [370, 307], [355], [270, 380], [322, 362], [308, 36], [382, 359], [277, 256, 36], [96, 79, 120], [262], [36, 306], [], [269], [361, 276, 322, 300, 340]]

def flatten_list(input_list: list[list]):
    """ Turns lists of lists into a single list """
    flattened_list = []
    for array in input_list:
        for x in array:
            flattened_list.append(x)
    return flattened_list

def plot_pixels(snrs, pixels, peaks):
        """ Plot the snr vs # pixels """
        
        # Initialise arrays
        blue_points_x = [pixels[0]]
        blue_points_y = [snrs[0]]
        green_points_x = []
        green_points_y = []
        orange_points_x = []
        orange_points_y = []
        red_points_x = []
        red_points_y = []

        # Sort points
        for snr, pix, pk in zip(snrs[1:], pixels[1:], peaks[1:]):
            num_pks = len(pk)
            
            # If there are no lines
            if len(snr) == 0:
                continue
            
            # bad - negative peaks
            if any(pk) < 0 or len(pix) == 0:
                pix = np.zeros(len(pk)).tolist()
                green_points_x.append(pix)
                green_points_y.append(snr)
                continue
            
            # bad - no redshift or extreme redshift (z>15)
            if num_pks < 2 or num_pks > 4: 
                green_points_x.append(pix)
                green_points_y.append(snr)
                continue
            
            # Two peaks is a good sign its real
            if num_pks == 2:
                pk_diffs = np.abs(np.diff(pk)) # needs to be at least 200 channels  
                if pk_diffs > 200:
                    red_points_x.append(pix)
                    red_points_y.append(snr)
                else:
                    orange_points_x.append(pix)
                    orange_points_y.append(snr)
            
            # Check 3 or 4 peaks
            else:
                pk_diffs = np.abs(np.diff(pk))            
                pk_diffs = np.abs(np.diff(pk_diffs))
                pk_diffs = np.average(pk_diffs)
                if pk_diffs < 15:
                    red_points_x.append(pix)
                    red_points_y.append(snr)
                else:
                    orange_points_x.append(pix)
                    orange_points_y.append(snr)
        
        # The number of points irrespective of how many sslf lines found
        blue_points = len(blue_points_x)
        green_points = len(green_points_x)
        orange_points = len(orange_points_x)
        red_points = len(red_points_x)

        # Flatten lists of lists to one big list
        blue_points_x = flatten_list(blue_points_x)
        blue_points_y = flatten_list(blue_points_y)
        green_points_x = flatten_list(green_points_x)
        green_points_y = flatten_list(green_points_y)
        orange_points_x = flatten_list(orange_points_x)
        orange_points_y = flatten_list(orange_points_y)
        red_points_x = flatten_list(red_points_x)
        red_points_y = flatten_list(red_points_y)
        
        # Random points
        with open(f'snr_vs_pix.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            rows = zip(['Blue', 'Green', 'Yellow', 'Red'], 
                       [blue_points, green_points, orange_points, red_points])
            for row in rows:
                wr.writerow(row)
            wr.writerow([])

        # Make the plot
        plt.figure(figsize=(20,9))
        plt.scatter(blue_points_x, blue_points_y, s=60, marker='*', color='blue')
        plt.scatter(green_points_x, green_points_y, s=60, marker='X', color='green')
        plt.scatter(orange_points_x, orange_points_y, s=60, marker='D', color='darkorange')
        plt.scatter(red_points_x, red_points_y, s=60, marker='s', color='red')
        plt.title(f'No. Random Points = {len(snrs)-1}', fontsize=20)
        plt.xlabel('No. of Pixels', fontsize=20)
        plt.ylabel('SNR', fontsize=20)
        plt.legend(['Target', 'No significance', 'Low significance', 'High significance'])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('SNR vs Pix.png')
        plt.show()
    
plot_pixels(snrs, pixels, peaks)