import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, ttest_ind, mannwhitneyu
import statsmodels.api as sma
import statsmodels.stats as smst


# Test equality of variances
def var_chk(ser1, ser2):
    stats, bf = levene(ser1, ser2, center='median')
    return round(bf, 3)


# Test normality
def norm_chk(ser):
    norm = smst.diagnostic.lilliefors(ser, pvalmethod='approx')[1]
    return round(norm, 3)


# Test two series for variance equality and normality; choose and apply compartible test
def t_test(ser1, ser2, parameter):
    # Test the conditions
    norm1, norm2 = norm_chk(ser1), norm_chk(ser2)  # p-values of Lilliefors tests
    bf = var_chk(ser1, ser2)  # p-value of Brown-Forsithe test
    normality = norm1 >= 0.05 and norm2 >= 0.05  # both samples are normal
    var_eq = bf >= 0.05  # variances of samples are equal

    if var_eq and normality:
        t = ttest_ind(ser1, ser2)  # Student's t-test
        test = '2-сторонний t-тест Стьюдента'

    elif not var_eq and normality:
        t = ttest_ind(ser1, ser2, equal_var=False)  # Welch t-test
        test = '2-сторонний t-тест Уэлча'

    elif not normality:
        t = mannwhitneyu(ser1, ser2)  # Mann-Whitney test
        test = 'Тест Манна-Уитни'

    return [t[0], round(t[1], 3), test, norm1, norm2, bf]


# Plot the histograms and quantille-quantille
def norm_plot(ser1, ser2, parameter):
    group = ['_Здор', '_Бол']

    for s in range(2):
        ser = [ser1, ser2][s]

        # plot histogram of series
        plt.hist(ser)
        plt.savefig(parameter[:5] + group[s] + '_hist.png')
        plt.clf()

        # plot q-q plot of series
        fig = sma.qqplot(ser, fit=True, line="45")
        plt.savefig(parameter[:5] + group[s] + '_qq_plot.png')
        plt.clf()


# replace the nan values by median of series
def median_fill(ser):
    m = ser.median()
    for s in range(len(ser)):
        if np.isnan(ser[s]):
            ser.loc[s] = m
    return ser


# Boxplots of vertical dataframe with one continuous variable (dependent) and two categorical variables (independent)

# value – is the column name with continuous variable values
# parameter – is the column name with names of groups or variables (for example 'weight', 'height' and 'age')
# category – is the column name with two gradations (for example 'healthy' and 'disease')

def boxplot(df, value, parameter, category):
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(6, 8))

    # Plot with vertical boxes
    sns.boxplot(data=df, x=parameter, y=value, hue=category,

                orient='v', dodge=True, whis=[2.5, 97.5], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(data=df, x=parameter, y=value, hue=category, orient='v', size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    plt.grid(True)
    plt.xticks(rotation=0)
    ax.set(ylabel="Величина показателя", xlabel="")
    sns.despine(trim=True, left=True)
    plt.savefig('boxplot.png')
