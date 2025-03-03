{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "#!pip install openpyxl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_ = pd.read_excel('data_3.xlsx')  #importing excel sheet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.792464</td>\n",
       "      <td>5.465294</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.495052</td>\n",
       "      <td>4.089470</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5.161569</td>\n",
       "      <td>5.407030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6.425300</td>\n",
       "      <td>6.453381</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.888931</td>\n",
       "      <td>4.667882</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          x         y\n",
       "0  4.792464  5.465294\n",
       "1  4.495052  4.089470\n",
       "2  5.161569  5.407030\n",
       "3  6.425300  6.453381\n",
       "4  4.888931  4.667882"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finding the no. of rows and columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 500 entries, 0 to 499\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   x       500 non-null    float64\n",
      " 1   y       500 non-null    float64\n",
      "dtypes: float64(2)\n",
      "memory usage: 7.9 KB\n"
     ]
    }
   ],
   "source": [
    "df_.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'y')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#visualizing the dataset\n",
    "plt.scatter(x= df_['x'] , y= df_['y'])  \n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Sampling of data\n",
    "import random\n",
    "df1 = df_.sample(n=100)      #Sample of size = 100\n",
    "df3 = df_.sample(n=100)      #Sample of size = 100\n",
    "df2 = df_.sample(n=150)      #Sample of size = 150\n",
    "df4 = df_.sample(n=150)      #Sample of size = 150\n",
    "df5 = df_.sample(n=475)      #Sample of size = 475"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The mean of population is: x: 5.082467729231954, y:4.9528964245044955\n",
      "The mean of df1 is: x: 5.208605136661817, y:4.863003718352751\n",
      "The mean of df3 is: x: 5.131614650959456, y:4.817565428334228\n",
      "The mean of df2 is: x: 5.231185089292317, y:4.992586018337656\n",
      "The mean of df4 is: x: 4.966454321189514, y:4.756073618534653\n",
      "The mean of df5 is: x: 5.088317580862573, y:4.971398303841467\n"
     ]
    }
   ],
   "source": [
    "#Comparing the mean of population and samples\n",
    "a, b = df_.mean()\n",
    "print('The mean of population is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df1.mean()\n",
    "print('The mean of df1 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df3.mean()\n",
    "print('The mean of df3 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df2.mean()\n",
    "print('The mean of df2 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df4.mean()\n",
    "print('The mean of df4 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df5.mean()\n",
    "print('The mean of df5 is: x: {}, y:{}'.format(a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The median of population is: x: 5.05487549200244, y:4.978847078474789\n",
      "The median of df1 is: x: 5.3577998306052805, y:4.693163419488252\n",
      "The median of df2 is: x: 5.194553256961091, y:4.934270539683137\n",
      "The median of df5 is: x: 5.064640223319679, y:5.013365520710948\n"
     ]
    }
   ],
   "source": [
    "#Comparing the median of population and samples\n",
    "a, b = df_.median()\n",
    "print('The median of population is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df1.median()\n",
    "print('The median of df1 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df2.median()\n",
    "print('The median of df2 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df5.median()\n",
    "print('The median of df5 is: x: {}, y:{}'.format(a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The variance of population is: x: 2.662558188756156, y:2.63475813398542\n",
      "The variance of df1 is: x: 2.509752099678357, y:3.0971560050963727\n",
      "The variance of df2 is: x: 2.6912241717142855, y:2.1507589531117413\n",
      "The variance of df3 is: x: 1.8105725934559411, y:2.33754858165354\n",
      "The variance of df4 is: x: 2.987878033357866, y:3.0109624255147414\n",
      "The variance of df5 is: x: 2.721158756053628, y:2.7047476389656757\n"
     ]
    }
   ],
   "source": [
    "#Comparing the variance of population and samples\n",
    "a, b = df_.var()\n",
    "print('The variance of population is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df1.var()\n",
    "print('The variance of df1 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df2.var()\n",
    "print('The variance of df2 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df3.var()\n",
    "print('The variance of df3 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df4.var()\n",
    "print('The variance of df4 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df5.var()\n",
    "print('The variance of df5 is: x: {}, y:{}'.format(a,b))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The standard deviation of population is: x: 1.6317347176413683, y:1.62319380666186\n",
      "The standard deviation of df1 is: x: 1.5842197131958549, y:1.7598738605639817\n",
      "The standard deviation of df2 is: x: 1.6404950995703356, y:1.4665466078893439\n",
      "The standard deviation of df3 is: x: 1.345575190561992, y:1.5289043729591265\n",
      "The standard deviation of df4 is: x: 1.7285479551802623, y:1.7352125015440447\n",
      "The standard deviation of df5 is: x: 1.6495935123701317, y:1.6446116985372796\n"
     ]
    }
   ],
   "source": [
    "#Comparing the standard deviation of population and samples\n",
    "a, b = df_.std()\n",
    "print('The standard deviation of population is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df1.std()\n",
    "print('The standard deviation of df1 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df2.std()\n",
    "print('The standard deviation of df2 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df3.std()\n",
    "print('The standard deviation of df3 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df4.std()\n",
    "print('The standard deviation of df4 is: x: {}, y:{}'.format(a,b))\n",
    "a, b = df5.std()\n",
    "print('The standard deviation of df5 is: x: {}, y:{}'.format(a,b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Statistics of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13.725427048471568"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Range of x\n",
    "popu_range_x = df_['x'].max() - df_['x'].min()\n",
    "popu_range_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11.593352800114099"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#range of y\n",
    "popu_range_y = df_['y'].max() - df_['y'].min()\n",
    "popu_range_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "max y: 10.58925243952102\n",
      "min y: -1.004100360593079\n",
      "max x: 12.2670245277286\n",
      "min x: -1.458402520742969\n"
     ]
    }
   ],
   "source": [
    "maxy=df_['y'].max()\n",
    "miny=df_['y'].min()\n",
    "print('max y: {}'.format(maxy))\n",
    "print('min y: {}'.format(miny))\n",
    "maxx=df_['x'].max()\n",
    "minx=df_['x'].min()\n",
    "print('max x: {}'.format(maxx))\n",
    "print('min x: {}'.format(minx))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0.10</th>\n",
       "      <td>3.435130</td>\n",
       "      <td>3.284788</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0.25</th>\n",
       "      <td>4.318981</td>\n",
       "      <td>4.218101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0.50</th>\n",
       "      <td>5.054875</td>\n",
       "      <td>4.978847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0.75</th>\n",
       "      <td>5.891603</td>\n",
       "      <td>5.782668</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0.90</th>\n",
       "      <td>6.533381</td>\n",
       "      <td>6.592286</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1.00</th>\n",
       "      <td>12.267025</td>\n",
       "      <td>10.589252</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              x          y\n",
       "0.10   3.435130   3.284788\n",
       "0.25   4.318981   4.218101\n",
       "0.50   5.054875   4.978847\n",
       "0.75   5.891603   5.782668\n",
       "0.90   6.533381   6.592286\n",
       "1.00  12.267025  10.589252"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#percentile calculation\n",
    "df_.quantile([0.10, 0.25, 0.5, 0.75,0.90, 1], axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Boxplot of Y')"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Quartile distribution using boxplot\n",
    "plt.figure(figsize=(12,4))\n",
    "plt.subplot(1,2,1)\n",
    "plt.boxplot(df_['x'])\n",
    "plt.xlabel('X')\n",
    "plt.title(\"Boxplot of X\")\n",
    "plt.subplot(1,2,2)\n",
    "plt.boxplot(df_['y'])\n",
    "plt.xlabel('Y')\n",
    "plt.title(\"Boxplot of Y\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Distribution of y')"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Visualizing histogram distribution of dataset\n",
    "\n",
    "plt.figure(figsize=(12,4))   \n",
    "plt.subplot(1,2,1)\n",
    "df_['x'].plot.hist(bins=50, alpha= 0.5, color='orange' )\n",
    "plt.title('Distribution of x')\n",
    "plt.xlabel('x')\n",
    "\n",
    "\n",
    "plt.subplot(1,2,2)\n",
    "df_['y'].plot.hist(bins=50, alpha= 0.5, color='purple' )\n",
    "plt.xlabel('y')\n",
    "plt.title('Distribution of y')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Skewness"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src = skewness.png>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "x    0.041523\n",
       "y   -0.082802\n",
       "dtype: float64"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#calculating skewness\n",
    "df_.skew()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kurtosis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src = kurtosis.jpg>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "x    3.346334\n",
       "y    2.454247\n",
       "dtype: float64"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Calculating kurtosis\n",
    "df_.kurt()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Grouped Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "#grp = pd.DataFrame({'Min':[4.1, 4.6, 5.1, 5.6, 6.1, 6.6], 'Max':[4.5,5.0,5.5,6.0,6.5,7.0], 'frequency' :[8, 4, 10, 6, 7, 5]}, index = '1 2 3 4 5 6'.split())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "#grp['mid point'] = (grp['Min']+grp['Max'])/2\n",
    "#grp['Cumulative Frequency'] = grp['frequency'].cumsum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "#grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "#calculating mean \n",
    "\n",
    "#mean = ((grp['frequency']*grp['mid point']).sum(axis=0))/(grp['frequency'].sum(axis=0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'mean' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[59], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m mean\n",
      "\u001b[0;31mNameError\u001b[0m: name 'mean' is not defined"
     ]
    }
   ],
   "source": [
    "mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "#calculating median\n",
    "#L = grp[grp['Cumulative Frequency']== 22]['Min']\n",
    "#n_2 = grp['frequency'].sum(axis=0)/2\n",
    "#f = grp[grp['Cumulative Frequency']== 22]['frequency']\n",
    "#c = 12\n",
    "#h = 0.5\n",
    "#median = ((n_2 - c)*h/f)+L\n",
    "#print('Median is {}'.format(median.values))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Outliers Detection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Standard Deviation Approach"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_arr= np.array(df_['x'])   #converting to array "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "def outliers(k):\n",
    "\n",
    "    outlier=[]\n",
    "\n",
    "    for i in range(0,500):\n",
    "        diff= (abs(x_arr[i]- np.mean(df_['x'])))/ np.std(df_['x'])\n",
    "        \n",
    "        if diff > k:\n",
    "            outlier.append(x_arr[i])\n",
    "            \n",
    "    print('Number of outliers in x: ',len(outlier))\n",
    "    ser= pd.Series(outlier, index=range(1,len(outlier)+1))\n",
    "    \n",
    "    ol=[]\n",
    "    \n",
    "    for i in range(0,500):\n",
    "        if x_arr[i] in outlier:\n",
    "            ol.append('1')\n",
    "        else:\n",
    "            ol.append('0')\n",
    "            \n",
    "    df_['outliers']=ol\n",
    "    sns.scatterplot(x='x',y='y',data=df_, hue='outliers')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of outliers in x:  96\n"
     ]
    }
   ],
   "source": [
    "k1 = outliers(1)  #For normally distributed data, k=1 represents 68.27% of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of outliers in x:  40\n"
     ]
    }
   ],
   "source": [
    "k2 = outliers(2)   #For normally distributed data, k=2 represents 95.45% of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of outliers in x:  10\n"
     ]
    }
   ],
   "source": [
    "k3 = outliers(3)    #For normally distributed data, k=3 represents 99.73% of the data\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Median Absolute Deviation (MAD) approach"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No. of outliers are:  33\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: title={'center': 'Distribution of y'}, xlabel='y', ylabel='Frequency'>"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Outliers using MAD approach\n",
    "\n",
    "out=[]\n",
    "count = 0\n",
    "X = df_.iloc[:,0]  #extracting x-values\n",
    "M = df_['x'].median()    #median of X\n",
    "z= abs(X-M)\n",
    "MAD= z.median()\n",
    "for i in range(0,500):\n",
    "    Zm= 0.6745*(X[i]-M)/MAD\n",
    "    if Zm>3 or Zm<-3:\n",
    "        out.append('1')\n",
    "        count += 1\n",
    "    else:\n",
    "        out.append('0')      \n",
    "df_['outliers']=out\n",
    "print ('No. of outliers are: ', count)\n",
    "sns.scatterplot(x='x',y='y',data=df_,hue='outliers')\n",
    "\n",
    "        \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Interquartile Approach"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No. of outliers are:  41\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: title={'center': 'Distribution of y'}, xlabel='y', ylabel='Frequency'>"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "outliers = []\n",
    "count = 0\n",
    "iqr = df_['x'].quantile(0.75) - df_['x'].quantile(0.25)\n",
    "a = df_['x'].quantile(0.75) + (1.5 * iqr)\n",
    "b = df_['x'].quantile(0.25) - (1.5 * iqr)\n",
    "for i in range(0,500):\n",
    "    Zm= df_.iloc[i,0]\n",
    "    if Zm>a or Zm<b:\n",
    "        outliers.append('1')\n",
    "        count += 1\n",
    "    else:\n",
    "        outliers.append('0')      \n",
    "df_['outliers']=outliers\n",
    "print ('No. of outliers are: ', count)\n",
    "sns.scatterplot(x='x',y='y',data=df_,hue='outliers')\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Expectation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5\n",
      "expectation is 2.799\n"
     ]
    }
   ],
   "source": [
    "##given\n",
    "x = [0, 1, 2, 3, 4]\n",
    "l = print(len(x))\n",
    "p = [0.008, 0.076, 0.265, 0.411, 0.240]\n",
    "e1= []\n",
    "for i in range(len(x)):\n",
    "    e2= x[i]*p[i]\n",
    "    e1.append(e2)\n",
    "Expected_value = sum(e1)\n",
    "print('expectation is', Expected_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Expectation is  3.56728703795613\n"
     ]
    },
    {
     "data": {
      "image/png": "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8b4t17W0tBTPP/88/P394eHhgWHDhiEtLc0xBQqysHTpUuHq6iq++uorcezYMTFp0iTh4eEhzp8/L3VpDdrAgQPFokWLxJEjR8TBgwfFkCFDRGRkpCgsLDTvM3fuXOHl5SVWrFghDh8+LEaOHClCQkKEVquVsPKGbe/evSI6OlrEx8eLSZMmmbfzWjtGTk6OiIqKEuPGjRN79uwRycnJYuPGjeLMmTPmfXitHePtt98Wfn5+4r///a9ITk4Wy5cvF56enuKjjz4y78NrbZ8//vhDvPnmm2LFihUCgFi1apXF+7Zc1/Hjx4uwsDCxYcMGceDAATFgwACRkJAg9Hp9retjULlB165dxfjx4y22tWrVSrz22msSVdQ4XblyRQAQ27ZtE0IIYTQaRXBwsJg7d655H51OJzQajfj888+lKrNBKygoEM2bNxcbNmwQ/fr1MwcVXmvHmTp1qujdu7fV93mtHWfIkCHiiSeesNh2//33i1GjRgkheK0d5cagYst1zcvLE66urmLp0qXmfS5evCjkcrlYt25drWti1891ysrKsH//ftx1110W2++66y7s3LlToqoap/z8fACAr68vACA5ORkZGRkW116lUqFfv3689naaMGEChgwZgjvuuMNiO6+146xZswadO3fGQw89hMDAQHTo0AFfffWV+X1ea8fp3bs3Nm3ahFOnTgEADh06hL///huDBw8GwGtdV2y5rvv370d5ebnFPqGhoWjXrp1Drn2Dfiiho2VlZcFgMCAoKMhie1BQEDIyMiSqqvERQmDy5Mno3bs32rVrBwDm61vVtT9//ny919jQLV26FAcOHMC+ffsqvcdr7Tjnzp3DwoULMXnyZLzxxhvYu3cvXnjhBahUKowZM4bX2oGmTp2K/Px8tGrVCgqFAgaDAbNnz8YjjzwCgL/XdcWW65qRkQGlUokmTZpU2scR904GlSrIZDKLr4UQlbaR/SZOnIikpCT8/fffld7jta+91NRUTJo0CevXr4ebm5vV/Xita89oNKJz58545513AAAdOnTA0aNHsXDhQowZM8a8H6917S1btgw//vgjfv75Z7Rt2xYHDx7Eiy++iNDQUIwdO9a8H6913bDnujrq2rPr5zr+/v5QKBSVEuCVK1cqpUmyz/PPP481a9Zgy5YtCA8PN28PDg4GAF57B9i/fz+uXLmCTp06wcXF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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#expectation of rolling a dice N times\n",
    "\n",
    "N = 100 ##no.. of trials-1\n",
    "roll = np.zeros(N)\n",
    "\n",
    "for i in range(N):\n",
    "    roll[i] = np.random.randint(1, 7) #filling up the sample space\n",
    "  #  print(roll)\n",
    "expectation = np.zeros(N)\n",
    "\n",
    "for i in range(1, N):\n",
    "    expectation[i] = np.mean(roll[0:i])\n",
    "  #  print(expectation)\n",
    "\n",
    "\n",
    "expected_value = np.mean(expectation)\n",
    "\n",
    "print(\"Expectation is \", expected_value)\n",
    "plt.plot(expectation)\n",
    "plt.title(\"Expectation of the dice roll\")\n",
    "plt.xlabel(\"Number of Trials\")\n",
    "plt.ylabel(\"Expectation\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy as sc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "dice = sc.stats.randint(1,7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.16666666666666666"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dice.pmf(4) #probability of getting 4 when we roll a dice"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8333333333333334"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dice.cdf(5)  #probability of getting a number less than or equal to 5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Covariance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src = covariance.png>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#covariance between values\n",
    "\n",
    "data = {'A': [45,37,42,35,39],\n",
    "        'B': [38,31,26,28,33],\n",
    "        'C': [10,15,17,21,12]\n",
    "        }\n",
    "\n",
    "df = pd.DataFrame(data,columns=['A','B','C'])\n",
    "\n",
    "covMatrix = df.cov()\n",
    "sns.heatmap(covMatrix, annot=True, fmt='g')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 15.8    9.6  -12.  ]\n",
      " [  9.6   21.7  -17.25]\n",
      " [-12.   -17.25  18.5 ]]\n"
     ]
    }
   ],
   "source": [
    "A = [45,37,42,35,39]\n",
    "B = [38,31,26,28,33]\n",
    "C = [10,15,17,21,12]\n",
    "\n",
    "data = np.array([A,B,C])\n",
    "\n",
    "covMatrix = np.cov(data)\n",
    "print (covMatrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          A         B         C\n",
      "A  1.000000  0.518457 -0.701886\n",
      "B  0.518457  1.000000 -0.860941\n",
      "C -0.701886 -0.860941  1.000000\n"
     ]
    },
    {
     "data": {
      "image/png": "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oLfDgZfx+wF/xO8HXGxIMKqRvLw82e125ebmqlatWt4OB17G7wf8Eb8fcLFhDQYAALAcCQYAALAcCQYAALAcCUY1EhgYqFdeeYUFXJDE7we44/cDLjYs8gQAAJajggEAACxHggEAACxHggEAACxHggEAACxHggEAACxHglFNrF69Wr6+vurWrZu3Q4GXPfbYY7LZbK6rbt266tatm7777jtvhwYvyc7O1uDBg9WoUSMFBgbq+uuv1913361ly5Z5OzSgXCQY1cS0adM0ePBgffvtt8rKyvJ2OPCybt26yeFwyOFwaNmyZfLz81PPnj29HRa8YM+ePYqMjNTy5cv1xhtvaMuWLUpNTVXnzp311FNPeTs8oFycg1ENnDx5UmFhYVq/fr1eeeUVNW/eXKNGjfJ2WPCSxx57TDk5OVqwYIGrbdWqVerUqZMOHTqka665xnvB4YLr0aOHvvvuO+3YsUM1a9Z0+y4nJ0e1a9f2TmDAOVDBqAbmzJmjJk2aqEmTJurfv78+/vhjkffhNydOnNCsWbN00003qW7dut4OBxfQr7/+qtTUVD311FOlkgtJJBeo1vy8HQCkqVOnqn///pLOlsZPnDihZcuWqUuXLl6ODN7yxRdf6Morr5T0e4Xriy++kI8Pfye4nOzatUtOp1NNmzb1dihAlfFfKy/bsWOH1q1bp4ceekiS5Ofnp759+2ratGlejgze1LlzZ2VmZiozM1Nr165VbGysunfvrl9++cXboeEC+q2SabPZvBwJUHVUMLxs6tSpOnPmjK699lpXm9PplL+/v44dO6Y6dep4MTp4S82aNXXTTTe5PkdGRsput2vKlCkaN26cFyPDhdS4cWPZbDZt27ZNvXr18nY4QJVQwfCiM2fOaMaMGXrrrbdcf1vNzMzU5s2bFR4erlmzZnk7RFQTNptNPj4+ys/P93YouICuvvpqde3aVe+9955OnjxZ6vucnJwLHxRQSSQYXvTFF1/o2LFjGjBggCIiItyuBx54QFOnTvV2iPCSgoICZWdnKzs7W9u2bdPgwYN14sQJ3X333d4ODRfYpEmTVFxcrHbt2mnu3LnauXOntm3bpokTJyomJsbb4QHlIsHwoqlTp6pLly6y2+2lvuvdu7cyMzO1adMmL0QGb0tNTVVYWJjCwsIUFRWl9evX67PPPtNtt93m7dBwgTVs2FCbNm1S586d9eyzzyoiIkJ33nmnli1bpvfff9/b4QHl4hwMAABgOSoYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAciQYAADAcv8fVomhcCb3BvYAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data = {'A': [45,37,42,35,39],\n",
    "        'B': [38,31,26,28,33],\n",
    "        'C': [10,15,17,21,12]\n",
    "        }\n",
    "\n",
    "df = pd.DataFrame(data,columns=['A','B','C'])\n",
    "\n",
    "corrMatrix = df.corr()\n",
    "\n",
    "print (corrMatrix)\n",
    "\n",
    "sns.heatmap(corrMatrix, annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Draw the samples from distribution "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Normal Distribution using scipy library\n",
    "## reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#:~:text=The%20location%20%28%20loc%29%20keyword%20specifies%20the%20mean.,them%20with%20details%20specific%20for%20this%20particular%20distribution.\n",
    "from scipy.stats import norm\n",
    "import matplotlib.pyplot as pyplot\n",
    "from scipy.stats import norm\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "#Norm_Dist= norm(loc=3, scale=5) # loc is mean, scale is standard deviation\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Samples from a Normal Distribution: [-0.42788176  1.45015423  0.71889254 ... -1.40346687 -0.25127813\n",
      " -0.55073386]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([1.0000e+02, 9.8700e+02, 6.4510e+03, 2.0918e+04, 3.3016e+04,\n",
       "        2.5957e+04, 1.0438e+04, 1.9190e+03, 2.0300e+02, 1.1000e+01]),\n",
       " array([-4.00795344, -3.14846272, -2.28897199, -1.42948126, -0.56999054,\n",
       "         0.28950019,  1.14899092,  2.00848165,  2.86797237,  3.7274631 ,\n",
       "         4.58695383]),\n",
       " <BarContainer object of 10 artists>)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjoAAAGdCAYAAAAbudkLAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAouElEQVR4nO3df0zUd57H8dcUZYosfBekwziRemTPclJsm2KDaLfVqqARqW1zusdlIncetucPlgCxZ/uHutmVbdW6l3p6tmnqrtrS3Lm23agcNG7pEsUfXElLa017axc8QfwxzggxA6Xf+6PnNx2xVvzRgY/PRzJJ5/t9z/D5Mrvl2S8zX1y2bdsCAAAw0B3RXgAAAMCtQugAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMNawaC8gmr7++mudPHlSCQkJcrlc0V4OAAC4BrZt68KFC/L5fLrjjqufs7mtQ+fkyZNKS0uL9jIAAMB1aGtr0+jRo686c1uHTkJCgqRvvlGJiYlRXg0AALgWoVBIaWlpzs/xq7mtQ+fSr6sSExMJHQAAhphredsJb0YGAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxhkV7AQCGhtWu1dFewoCttFdGewkAoowzOgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjDWg0Nm8ebPuu+8+JSYmKjExUbm5udq7d6+z37ZtrVq1Sj6fT3FxcZoyZYo++eSTiOcIh8NatmyZUlJSFB8fr8LCQp04cSJiJhAIyO/3y7IsWZYlv9+v8+fPR8y0trZqzpw5io+PV0pKikpLS9XT0zPAwwcAACYbUOiMHj1av/71r3XkyBEdOXJEjz32mB5//HEnZl588UW99NJL2rhxow4fPiyv16sZM2bowoULznOUlZVp165dqq6uVkNDg7q6ulRQUKC+vj5npqioSM3NzaqpqVFNTY2am5vl9/ud/X19fZo9e7a6u7vV0NCg6upq7dy5UxUVFTf6/QAAAAZx2bZt38gTJCcna+3atfrHf/xH+Xw+lZWV6dlnn5X0zdmb1NRUvfDCC3r66acVDAZ11113adu2bZo/f74k6eTJk0pLS9OePXuUn5+vo0ePKjMzU42NjcrJyZEkNTY2Kjc3V5999pkyMjK0d+9eFRQUqK2tTT6fT5JUXV2t4uJidXZ2KjEx8ZrWHgqFZFmWgsHgNT8GuF2tdq2O9hIGbKW9MtpLAHALDOTn93W/R6evr0/V1dXq7u5Wbm6ujh8/ro6ODuXl5Tkzbrdbjz76qPbv3y9JampqUm9vb8SMz+dTVlaWM3PgwAFZluVEjiRNnDhRlmVFzGRlZTmRI0n5+fkKh8Nqamr6zjWHw2GFQqGIGwAAMNeAQ+fjjz/Wj370I7ndbj3zzDPatWuXMjMz1dHRIUlKTU2NmE9NTXX2dXR0KDY2VklJSVed8Xg8/b6ux+OJmLn86yQlJSk2NtaZuZKqqirnfT+WZSktLW2ARw8AAIaSAYdORkaGmpub1djYqH/+53/WggUL9Omnnzr7XS5XxLxt2/22Xe7ymSvNX8/M5VasWKFgMOjc2trarrouAAAwtA04dGJjY/XXf/3XmjBhgqqqqnT//ffrX//1X+X1eiWp3xmVzs5O5+yL1+tVT0+PAoHAVWdOnTrV7+uePn06YubyrxMIBNTb29vvTM+3ud1u5xNjl24AAMBcN3wdHdu2FQ6HlZ6eLq/Xq7q6OmdfT0+P6uvrNWnSJElSdna2hg8fHjHT3t6ulpYWZyY3N1fBYFCHDh1yZg4ePKhgMBgx09LSovb2dmemtrZWbrdb2dnZN3pIAADAEMMGMvzcc89p1qxZSktL04ULF1RdXa33339fNTU1crlcKisr05o1azR27FiNHTtWa9as0YgRI1RUVCRJsixLCxcuVEVFhUaOHKnk5GRVVlZq/Pjxmj59uiRp3LhxmjlzpkpKSrRlyxZJ0qJFi1RQUKCMjAxJUl5enjIzM+X3+7V27VqdO3dOlZWVKikp4SwNAABwDCh0Tp06Jb/fr/b2dlmWpfvuu081NTWaMWOGJGn58uW6ePGiFi9erEAgoJycHNXW1iohIcF5jg0bNmjYsGGaN2+eLl68qGnTpmnr1q2KiYlxZnbs2KHS0lLn01mFhYXauHGjsz8mJka7d+/W4sWLNXnyZMXFxamoqEjr1q27oW8GAAAwyw1fR2co4zo6wLXjOjoABosf5Do6AAAAgx2hAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMNawaC8AAG6V1a7V0V7CgK20V0Z7CYBROKMDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIw1oNCpqqrSQw89pISEBHk8Hs2dO1fHjh2LmCkuLpbL5Yq4TZw4MWImHA5r2bJlSklJUXx8vAoLC3XixImImUAgIL/fL8uyZFmW/H6/zp8/HzHT2tqqOXPmKD4+XikpKSotLVVPT89ADgkAABhsQKFTX1+vJUuWqLGxUXV1dfrqq6+Ul5en7u7uiLmZM2eqvb3due3Zsydif1lZmXbt2qXq6mo1NDSoq6tLBQUF6uvrc2aKiorU3Nysmpoa1dTUqLm5WX6/39nf19en2bNnq7u7Ww0NDaqurtbOnTtVUVFxPd8HAABgoAFdR6empibi/uuvvy6Px6OmpiY98sgjzna32y2v13vF5wgGg3rttde0bds2TZ8+XZK0fft2paWl6b333lN+fr6OHj2qmpoaNTY2KicnR5L06quvKjc3V8eOHVNGRoZqa2v16aefqq2tTT6fT5K0fv16FRcX61e/+pUSExMHcmgAAMBAN/QenWAwKElKTk6O2P7+++/L4/HonnvuUUlJiTo7O519TU1N6u3tVV5enrPN5/MpKytL+/fvlyQdOHBAlmU5kSNJEydOlGVZETNZWVlO5EhSfn6+wuGwmpqarrjecDisUCgUcQMAAOa67tCxbVvl5eV6+OGHlZWV5WyfNWuWduzYoX379mn9+vU6fPiwHnvsMYXDYUlSR0eHYmNjlZSUFPF8qamp6ujocGY8Hk+/r+nxeCJmUlNTI/YnJSUpNjbWmblcVVWV854fy7KUlpZ2vYcPAACGgOv+ExBLly7VRx99pIaGhojt8+fPd/45KytLEyZM0JgxY7R79249+eST3/l8tm3L5XI597/9zzcy820rVqxQeXm5cz8UChE7AAAY7LrO6Cxbtkzvvvuu/vjHP2r06NFXnR01apTGjBmjzz//XJLk9XrV09OjQCAQMdfZ2emcofF6vTp16lS/5zp9+nTEzOVnbgKBgHp7e/ud6bnE7XYrMTEx4gYAAMw1oNCxbVtLly7V73//e+3bt0/p6enf+5izZ8+qra1No0aNkiRlZ2dr+PDhqqurc2ba29vV0tKiSZMmSZJyc3MVDAZ16NAhZ+bgwYMKBoMRMy0tLWpvb3dmamtr5Xa7lZ2dPZDDAgAAhhrQr66WLFmiN954Q++8844SEhKcMyqWZSkuLk5dXV1atWqVnnrqKY0aNUpffvmlnnvuOaWkpOiJJ55wZhcuXKiKigqNHDlSycnJqqys1Pjx451PYY0bN04zZ85USUmJtmzZIklatGiRCgoKlJGRIUnKy8tTZmam/H6/1q5dq3PnzqmyslIlJSWcqQEAAJIGeEZn8+bNCgaDmjJlikaNGuXc3nrrLUlSTEyMPv74Yz3++OO65557tGDBAt1zzz06cOCAEhISnOfZsGGD5s6dq3nz5mny5MkaMWKE/vCHPygmJsaZ2bFjh8aPH6+8vDzl5eXpvvvu07Zt25z9MTEx2r17t+68805NnjxZ8+bN09y5c7Vu3bob/Z4AAABDuGzbtqO9iGgJhUKyLEvBYJCzQMD3WO1aHe0l3BZW2iujvQRg0BvIz2/+1hUAADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYw6K9AOB2tNq1OtpLAIDbAmd0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxBhQ6VVVVeuihh5SQkCCPx6O5c+fq2LFjETO2bWvVqlXy+XyKi4vTlClT9Mknn0TMhMNhLVu2TCkpKYqPj1dhYaFOnDgRMRMIBOT3+2VZlizLkt/v1/nz5yNmWltbNWfOHMXHxyslJUWlpaXq6ekZyCEBAACDDSh06uvrtWTJEjU2Nqqurk5fffWV8vLy1N3d7cy8+OKLeumll7Rx40YdPnxYXq9XM2bM0IULF5yZsrIy7dq1S9XV1WpoaFBXV5cKCgrU19fnzBQVFam5uVk1NTWqqalRc3Oz/H6/s7+vr0+zZ89Wd3e3GhoaVF1drZ07d6qiouJGvh8AAMAgLtu27et98OnTp+XxeFRfX69HHnlEtm3L5/OprKxMzz77rKRvzt6kpqbqhRde0NNPP61gMKi77rpL27Zt0/z58yVJJ0+eVFpamvbs2aP8/HwdPXpUmZmZamxsVE5OjiSpsbFRubm5+uyzz5SRkaG9e/eqoKBAbW1t8vl8kqTq6moVFxers7NTiYmJ37v+UCgky7IUDAavaR64WbiODr7LSntltJcADHoD+fl9Q+/RCQaDkqTk5GRJ0vHjx9XR0aG8vDxnxu1269FHH9X+/fslSU1NTert7Y2Y8fl8ysrKcmYOHDggy7KcyJGkiRMnyrKsiJmsrCwnciQpPz9f4XBYTU1NV1xvOBxWKBSKuAEAAHNdd+jYtq3y8nI9/PDDysrKkiR1dHRIklJTUyNmU1NTnX0dHR2KjY1VUlLSVWc8Hk+/r+nxeCJmLv86SUlJio2NdWYuV1VV5bznx7IspaWlDfSwAQDAEHLdobN06VJ99NFHevPNN/vtc7lcEfdt2+637XKXz1xp/npmvm3FihUKBoPOra2t7aprAgAAQ9t1hc6yZcv07rvv6o9//KNGjx7tbPd6vZLU74xKZ2enc/bF6/Wqp6dHgUDgqjOnTp3q93VPnz4dMXP51wkEAurt7e13pucSt9utxMTEiBsAADDXgELHtm0tXbpUv//977Vv3z6lp6dH7E9PT5fX61VdXZ2zraenR/X19Zo0aZIkKTs7W8OHD4+YaW9vV0tLizOTm5urYDCoQ4cOOTMHDx5UMBiMmGlpaVF7e7szU1tbK7fbrezs7IEcFgAAMNSA/nr5kiVL9MYbb+idd95RQkKCc0bFsizFxcXJ5XKprKxMa9as0dixYzV27FitWbNGI0aMUFFRkTO7cOFCVVRUaOTIkUpOTlZlZaXGjx+v6dOnS5LGjRunmTNnqqSkRFu2bJEkLVq0SAUFBcrIyJAk5eXlKTMzU36/X2vXrtW5c+dUWVmpkpISztQAAABJAwydzZs3S5KmTJkSsf31119XcXGxJGn58uW6ePGiFi9erEAgoJycHNXW1iohIcGZ37Bhg4YNG6Z58+bp4sWLmjZtmrZu3aqYmBhnZseOHSotLXU+nVVYWKiNGzc6+2NiYrR7924tXrxYkydPVlxcnIqKirRu3boBfQMAAIC5bug6OkMd19FBtHAdHXwXrqMDfL8f7Do6AAAAgxmhAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMNeDQ+eCDDzRnzhz5fD65XC69/fbbEfuLi4vlcrkibhMnToyYCYfDWrZsmVJSUhQfH6/CwkKdOHEiYiYQCMjv98uyLFmWJb/fr/Pnz0fMtLa2as6cOYqPj1dKSopKS0vV09Mz0EMCAACGGnDodHd36/7779fGjRu/c2bmzJlqb293bnv27InYX1ZWpl27dqm6uloNDQ3q6upSQUGB+vr6nJmioiI1NzerpqZGNTU1am5ult/vd/b39fVp9uzZ6u7uVkNDg6qrq7Vz505VVFQM9JAAAIChhg30AbNmzdKsWbOuOuN2u+X1eq+4LxgM6rXXXtO2bds0ffp0SdL27duVlpam9957T/n5+Tp69KhqamrU2NionJwcSdKrr76q3NxcHTt2TBkZGaqtrdWnn36qtrY2+Xw+SdL69etVXFysX/3qV0pMTBzooQEAAMPckvfovP/++/J4PLrnnntUUlKizs5OZ19TU5N6e3uVl5fnbPP5fMrKytL+/fslSQcOHJBlWU7kSNLEiRNlWVbETFZWlhM5kpSfn69wOKympqYrriscDisUCkXcAACAuW566MyaNUs7duzQvn37tH79eh0+fFiPPfaYwuGwJKmjo0OxsbFKSkqKeFxqaqo6OjqcGY/H0++5PR5PxExqamrE/qSkJMXGxjozl6uqqnLe82NZltLS0m74eAEAwOA14F9dfZ/58+c7/5yVlaUJEyZozJgx2r17t5588snvfJxt23K5XM79b//zjcx824oVK1ReXu7cD4VCxA4AAAa75R8vHzVqlMaMGaPPP/9ckuT1etXT06NAIBAx19nZ6Zyh8Xq9OnXqVL/nOn36dMTM5WduAoGAent7+53pucTtdisxMTHiBgAAzHXLQ+fs2bNqa2vTqFGjJEnZ2dkaPny46urqnJn29na1tLRo0qRJkqTc3FwFg0EdOnTImTl48KCCwWDETEtLi9rb252Z2tpaud1uZWdn3+rDAgAAQ8CAf3XV1dWlL774wrl//PhxNTc3Kzk5WcnJyVq1apWeeuopjRo1Sl9++aWee+45paSk6IknnpAkWZalhQsXqqKiQiNHjlRycrIqKys1fvx451NY48aN08yZM1VSUqItW7ZIkhYtWqSCggJlZGRIkvLy8pSZmSm/36+1a9fq3LlzqqysVElJCWdqAACApOsInSNHjmjq1KnO/UvveVmwYIE2b96sjz/+WL/73e90/vx5jRo1SlOnTtVbb72lhIQE5zEbNmzQsGHDNG/ePF28eFHTpk3T1q1bFRMT48zs2LFDpaWlzqezCgsLI67dExMTo927d2vx4sWaPHmy4uLiVFRUpHXr1g38uwAAAIzksm3bjvYioiUUCsmyLAWDQc4C4Qe12rU62kvAILXSXhntJQCD3kB+fvO3rgAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABiL0AEAAMYidAAAgLEIHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgrGEDfcAHH3ygtWvXqqmpSe3t7dq1a5fmzp3r7LdtW6tXr9Yrr7yiQCCgnJwc/du//ZvuvfdeZyYcDquyslJvvvmmLl68qGnTpmnTpk0aPXq0MxMIBFRaWqp3331XklRYWKiXX35ZP/7xj52Z1tZWLVmyRPv27VNcXJyKioq0bt06xcbGXse3AgCib7VrdbSXMGAr7ZXRXgLwnQZ8Rqe7u1v333+/Nm7ceMX9L774ol566SVt3LhRhw8fltfr1YwZM3ThwgVnpqysTLt27VJ1dbUaGhrU1dWlgoIC9fX1OTNFRUVqbm5WTU2Nampq1NzcLL/f7+zv6+vT7Nmz1d3drYaGBlVXV2vnzp2qqKgY6CEBAABDuWzbtq/7wS5XxBkd27bl8/lUVlamZ599VtI3Z29SU1P1wgsv6Omnn1YwGNRdd92lbdu2af78+ZKkkydPKi0tTXv27FF+fr6OHj2qzMxMNTY2KicnR5LU2Nio3NxcffbZZ8rIyNDevXtVUFCgtrY2+Xw+SVJ1dbWKi4vV2dmpxMTE711/KBSSZVkKBoPXNA/cLEPxv9qB78IZHfzQBvLz+6a+R+f48ePq6OhQXl6es83tduvRRx/V/v37JUlNTU3q7e2NmPH5fMrKynJmDhw4IMuynMiRpIkTJ8qyrIiZrKwsJ3IkKT8/X+FwWE1NTTfzsAAAwBA14PfoXE1HR4ckKTU1NWJ7amqq/vKXvzgzsbGxSkpK6jdz6fEdHR3yeDz9nt/j8UTMXP51kpKSFBsb68xcLhwOKxwOO/dDodBADg8AAAwxt+RTVy6XK+K+bdv9tl3u8pkrzV/PzLdVVVXJsiznlpaWdtU1AQCAoe2mho7X65WkfmdUOjs7nbMvXq9XPT09CgQCV505depUv+c/ffp0xMzlXycQCKi3t7ffmZ5LVqxYoWAw6Nza2tqu4ygBAMBQcVNDJz09XV6vV3V1dc62np4e1dfXa9KkSZKk7OxsDR8+PGKmvb1dLS0tzkxubq6CwaAOHTrkzBw8eFDBYDBipqWlRe3t7c5MbW2t3G63srOzr7g+t9utxMTEiBsAADDXgN+j09XVpS+++MK5f/z4cTU3Nys5OVl33323ysrKtGbNGo0dO1Zjx47VmjVrNGLECBUVFUmSLMvSwoULVVFRoZEjRyo5OVmVlZUaP368pk+fLkkaN26cZs6cqZKSEm3ZskWStGjRIhUUFCgjI0OSlJeXp8zMTPn9fq1du1bnzp1TZWWlSkpKCBgAACDpOkLnyJEjmjp1qnO/vLxckrRgwQJt3bpVy5cv18WLF7V48WLngoG1tbVKSEhwHrNhwwYNGzZM8+bNcy4YuHXrVsXExDgzO3bsUGlpqfPprMLCwohr98TExGj37t1avHixJk+eHHHBQAAAAOkGr6Mz1HEdHUQL19GBSbiODn5oUbuODgAAwGBC6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAw1rBoLwC4Uatdq6O9BADAIMUZHQAAYCxCBwAAGIvQAQAAxiJ0AACAsQgdAABgLEIHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABjrpofOqlWr5HK5Im5er9fZb9u2Vq1aJZ/Pp7i4OE2ZMkWffPJJxHOEw2EtW7ZMKSkpio+PV2FhoU6cOBExEwgE5Pf7ZVmWLMuS3+/X+fPnb/bhAACAIeyWnNG599571d7e7tw+/vhjZ9+LL76ol156SRs3btThw4fl9Xo1Y8YMXbhwwZkpKyvTrl27VF1drYaGBnV1damgoEB9fX3OTFFRkZqbm1VTU6Oamho1NzfL7/ffisMBAABD1C35o57Dhg2LOItziW3b+s1vfqPnn39eTz75pCTpt7/9rVJTU/XGG2/o6aefVjAY1GuvvaZt27Zp+vTpkqTt27crLS1N7733nvLz83X06FHV1NSosbFROTk5kqRXX31Vubm5OnbsmDIyMm7FYQEAgCHmlpzR+fzzz+Xz+ZSenq6f/exn+vOf/yxJOn78uDo6OpSXl+fMut1uPfroo9q/f78kqampSb29vREzPp9PWVlZzsyBAwdkWZYTOZI0ceJEWZblzFxJOBxWKBSKuAEAAHPd9NDJycnR7373O/3Xf/2XXn31VXV0dGjSpEk6e/asOjo6JEmpqakRj0lNTXX2dXR0KDY2VklJSVed8Xg8/b62x+NxZq6kqqrKeU+PZVlKS0u7oWMFAACD200PnVmzZumpp57S+PHjNX36dO3evVvSN7+iusTlckU8xrbtftsud/nMlea/73lWrFihYDDo3Nra2q7pmAAAwNB0yz9eHh8fr/Hjx+vzzz933rdz+VmXzs5O5yyP1+tVT0+PAoHAVWdOnTrV72udPn2639mib3O73UpMTIy4AQAAc93y0AmHwzp69KhGjRql9PR0eb1e1dXVOft7enpUX1+vSZMmSZKys7M1fPjwiJn29na1tLQ4M7m5uQoGgzp06JAzc/DgQQWDQWcGAADgpn/qqrKyUnPmzNHdd9+tzs5O/fKXv1QoFNKCBQvkcrlUVlamNWvWaOzYsRo7dqzWrFmjESNGqKioSJJkWZYWLlyoiooKjRw5UsnJyaqsrHR+FSZJ48aN08yZM1VSUqItW7ZIkhYtWqSCggI+cQUAABw3PXROnDihv/u7v9OZM2d01113aeLEiWpsbNSYMWMkScuXL9fFixe1ePFiBQIB5eTkqLa2VgkJCc5zbNiwQcOGDdO8efN08eJFTZs2TVu3blVMTIwzs2PHDpWWljqfziosLNTGjRtv9uEAAIAhzGXbth3tRURLKBSSZVkKBoO8X2cIW+1aHe0lALe1lfbKaC8Bt5mB/Pzmb10BAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjEToAAMBYw6K9AADA0LbatTraSxiwlfbKaC8BPxDO6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAwFqEDAACMRegAAABjcWVkRBiKVzgFAOC7cEYHAAAYi9ABAADGInQAAICxCB0AAGAsQgcAABhryIfOpk2blJ6erjvvvFPZ2dn605/+FO0lAQCAQWJIh85bb72lsrIyPf/88/rwww/105/+VLNmzVJra2u0lwYAAAYBl23bdrQXcb1ycnL04IMPavPmzc62cePGae7cuaqqqvrex4dCIVmWpWAwqMTExFu51CGD6+gAwOC00l4Z7SUMGgP5+T1kLxjY09OjpqYm/cu//EvE9ry8PO3fv/+KjwmHwwqHw879YDAo6Ztv2K1QZX1/bAEAcC1u1c+qoejS9+JaztUM2dA5c+aM+vr6lJqaGrE9NTVVHR0dV3xMVVWVVq/uf8YiLS3tlqwRAICb5dfWr6O9hEHnwoULsizrqjNDNnQucblcEfdt2+637ZIVK1aovLzcuf/111/r3LlzGjly5Hc+5nqFQiGlpaWpra2NX4sNUrxGgx+v0eDHazT4mfga2batCxcuyOfzfe/skA2dlJQUxcTE9Dt709nZ2e8szyVut1tutzti249//ONbtURJUmJiojH/wzIVr9Hgx2s0+PEaDX6mvUbfdybnkiH7qavY2FhlZ2errq4uYntdXZ0mTZoUpVUBAIDBZMie0ZGk8vJy+f1+TZgwQbm5uXrllVfU2tqqZ555JtpLAwAAg8CQDp358+fr7Nmz+sUvfqH29nZlZWVpz549GjNmTLSXJrfbrZUrV/b7VRkGD16jwY/XaPDjNRr8bvfXaEhfRwcAAOBqhux7dAAAAL4PoQMAAIxF6AAAAGMROgAAwFiEzg8oHA7rgQcekMvlUnNzc7SXg//35ZdfauHChUpPT1dcXJx+8pOfaOXKlerp6Yn20m5rmzZtUnp6uu68805lZ2frT3/6U7SXhP9XVVWlhx56SAkJCfJ4PJo7d66OHTsW7WXhKqqqquRyuVRWVhbtpfzgCJ0f0PLly6/pctX4YX322Wf6+uuvtWXLFn3yySfasGGD/v3f/13PPfdctJd223rrrbdUVlam559/Xh9++KF++tOfatasWWptbY320iCpvr5eS5YsUWNjo+rq6vTVV18pLy9P3d3d0V4aruDw4cN65ZVXdN9990V7KVHBx8t/IHv37lV5ebl27type++9Vx9++KEeeOCBaC8L32Ht2rXavHmz/vznP0d7KbelnJwcPfjgg9q8ebOzbdy4cZo7d66qqqqiuDJcyenTp+XxeFRfX69HHnkk2svBt3R1denBBx/Upk2b9Mtf/lIPPPCAfvOb30R7WT8ozuj8AE6dOqWSkhJt27ZNI0aMiPZycA2CwaCSk5OjvYzbUk9Pj5qampSXlxexPS8vT/v374/SqnA1wWBQkvj/zCC0ZMkSzZ49W9OnT4/2UqJmSF8ZeSiwbVvFxcV65plnNGHCBH355ZfRXhK+x//8z//o5Zdf1vr166O9lNvSmTNn1NfX1++P86ampvb7I76IPtu2VV5erocfflhZWVnRXg6+pbq6Wv/93/+tw4cPR3spUcUZneu0atUquVyuq96OHDmil19+WaFQSCtWrIj2km871/oafdvJkyc1c+ZM/e3f/q3+6Z/+KUorhyS5XK6I+7Zt99uG6Fu6dKk++ugjvfnmm9FeCr6lra1NP//5z7V9+3bdeeed0V5OVPEenet05swZnTlz5qozf/VXf6Wf/exn+sMf/hDxL+i+vj7FxMTo7//+7/Xb3/72Vi/1tnWtr9GlfwmcPHlSU6dOVU5OjrZu3ao77uC/A6Khp6dHI0aM0H/8x3/oiSeecLb//Oc/V3Nzs+rr66O4OnzbsmXL9Pbbb+uDDz5Qenp6tJeDb3n77bf1xBNPKCYmxtnW19cnl8ulO+64Q+FwOGKfyQidW6y1tVWhUMi5f/LkSeXn5+s///M/lZOTo9GjR0dxdbjkf//3fzV16lRlZ2dr+/btt82/AAarnJwcZWdna9OmTc62zMxMPf7447wZeRCwbVvLli3Trl279P7772vs2LHRXhIuc+HCBf3lL3+J2PYP//AP+pu/+Rs9++yzt9WvGXmPzi129913R9z/0Y9+JEn6yU9+QuQMEidPntSUKVN09913a926dTp9+rSzz+v1RnFlt6/y8nL5/X5NmDBBubm5euWVV9Ta2qpnnnkm2kuDvnmD6xtvvKF33nlHCQkJznunLMtSXFxclFcHSUpISOgXM/Hx8Ro5cuRtFTkSoQOotrZWX3zxhb744ot+8ckJz+iYP3++zp49q1/84hdqb29XVlaW9uzZozFjxkR7aZCcj/1PmTIlYvvrr7+u4uLiH35BwFXwqysAAGAs3m0JAACMRegAAABjEToAAMBYhA4AADAWoQMAAIxF6AAAAGMROgAAwFiEDgAAMBahAwAAjEXoAAAAYxE6AADAWIQOAAAw1v8BhuBtNHRkuiQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Let's draw 50 samples from it and arrange as a 4 by 5 array\n",
    "#N_Dist = Norm_Dist.rvs(size=(1000)) # size will return the desired output shape or size\n",
    "N_Dist = np.random.normal(loc=0.0, scale=1.0, size=100000)   # loc is mean, scale is standard deviation, size = output size\n",
    "\n",
    "\n",
    "print(\"Samples from a Normal Distribution:\", N_Dist)\n",
    "pyplot.hist(N_Dist,color='purple')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "samples from T Distribution using scipy stats library are:  [ 8.72486016e-02 -7.58466097e-01 -5.23603135e-01  1.35689434e+00\n",
      " -1.15062722e+00 -3.40412878e-01 -1.39330400e+00 -1.74408790e+00\n",
      "  4.90994914e-02  1.54413476e+00 -2.69692774e-01 -6.31634621e-02\n",
      " -3.71814041e-02 -1.24022869e+00  9.55756479e-01  1.09459224e+00\n",
      "  1.14390368e+00 -1.79583403e-01  2.60659585e+00 -9.76655899e-01\n",
      " -1.30805155e+00  3.46083281e-01 -1.56962951e+00 -1.70326714e+00\n",
      "  2.09718052e+00 -1.30052423e+00  6.91237830e-01  5.77704467e-01\n",
      " -1.05646031e-01 -2.24591094e+00  5.92665929e-01 -1.42837731e+00\n",
      "  7.37514316e-01 -6.71367467e-01  2.24847777e+00 -1.04631792e+00\n",
      " -1.50073307e+00  1.21765270e+00  2.94762788e+00 -1.15147198e+00\n",
      "  8.49442743e-01 -1.54406286e-01  1.34972040e+00  1.07666362e+00\n",
      " -3.70689608e-01  7.22335428e-01 -3.16830091e-01  3.17906118e-01\n",
      "  7.51470048e-01 -7.88763676e-01 -7.75551972e-01 -1.38368924e-02\n",
      "  8.82683366e-01  1.95668148e+00  5.50329575e+00 -2.12245278e+00\n",
      "  3.94028171e-01  1.00686195e+00 -1.75105305e+00  2.98243282e-01\n",
      "  9.17302516e-01 -1.40594413e+00  1.48521856e+00 -1.39172928e+00\n",
      " -2.12193265e+00  5.68749874e-01  3.06951747e-01 -2.90166538e-01\n",
      "  1.38864415e+00 -4.97565286e-01  8.73864492e-01 -6.33902296e-01\n",
      " -3.72419713e-01 -9.59087583e-02  2.76157200e-02 -1.37027553e+00\n",
      "  6.38846755e-01  1.81951076e+00  3.38266284e-01 -3.44199595e-02\n",
      "  4.02410326e-02  4.06984680e-01 -2.49193275e-01  1.36173690e+00\n",
      " -2.87176718e+00 -1.07582092e+00  1.13598970e+00 -8.91412032e-01\n",
      " -7.15249610e-02 -2.14642105e+00 -3.61759561e-01 -1.71760568e+00\n",
      " -7.19926158e-01  3.44136565e-01  9.92238364e-01  1.23037950e+00\n",
      " -8.74779518e-01  1.75875261e+00  2.97402934e+00 -3.32296371e-01\n",
      "  6.78245669e-01  8.76987863e-01 -6.60491136e-01  5.72476913e-01\n",
      " -8.80640143e-01  9.77563352e-01  3.76499318e-01  1.10923777e+00\n",
      " -1.89024411e-01 -1.02782274e+00  3.07873843e-01  1.48904168e-01\n",
      " -5.43304636e-01  4.79462108e-01  1.85914198e+00 -9.33266951e-01\n",
      " -9.30211813e-01  1.15994504e+00  4.92255172e-01  1.38102351e+00\n",
      " -4.07259863e-01  1.07637201e+00 -1.69137695e+00 -3.57915819e-01\n",
      "  8.66299226e-01  1.03282916e-01  8.89746997e-01 -4.15160035e-01\n",
      " -1.66993547e-01  3.84047008e-02 -1.95451676e-01  4.57768203e-01\n",
      "  6.40690309e-01  2.68160017e-01 -5.74694952e-01 -1.26260828e+00\n",
      "  2.11588901e+00 -3.66796106e-01 -3.30641952e-01 -2.19873295e-01\n",
      "  9.73661682e-01 -8.18859750e-01  1.90846120e-01 -3.92443804e-01\n",
      " -9.27333779e-01  8.89080763e-01  1.25078070e+00 -1.77819559e+00\n",
      " -8.92832119e-01  1.65254569e+00  9.51905580e-01  1.23076483e+00\n",
      "  4.14900517e-01  1.45501178e+00 -4.71170841e-01  5.23728781e-01\n",
      " -1.45765564e+00  1.21255711e+00  3.20626914e-01 -7.52971040e-01\n",
      " -5.41786611e-01  2.74225122e+00 -1.30765266e-01  8.63838745e-01\n",
      " -2.83331789e+00  3.35238504e-01 -5.79706919e-01 -2.93718900e+00\n",
      "  3.08684202e-01 -1.78358019e+00 -1.99066982e+00 -1.37098522e+00\n",
      "  1.70480593e-01  5.71433926e-03  3.77754835e-01  1.76243502e+00\n",
      "  1.83435138e-01  7.15525010e-01 -1.06972667e+00  7.85861298e-01\n",
      " -2.19439773e-01  2.37886134e-01  4.58770391e-02  6.91637404e-01\n",
      "  1.04844374e-01 -2.19122411e+00  2.61720013e-01  3.92082709e-01\n",
      "  1.51286005e+00 -2.49164758e+00 -8.19536304e-01  6.42099072e-01\n",
      "  1.62504994e+00 -1.39168264e-01  1.59395962e-02  1.17435866e+00\n",
      " -1.40438307e+00 -1.58476668e+00 -4.79871141e-01  2.38497367e-01\n",
      "  4.71974803e-01  2.44674962e+00  2.84949891e+00 -8.08011752e-01\n",
      " -9.04849434e-02  4.13093162e-02  2.93066762e-01  1.22206087e+00\n",
      " -1.05014387e+00  1.11378728e-01  5.78527886e-01  4.68947626e-01\n",
      "  1.50152470e+00 -1.97863437e+00 -9.89255464e-01  1.14875654e-01\n",
      "  2.37298701e-01  1.02133637e+00  1.90640237e+00 -1.21191537e-01\n",
      "  4.03139261e-01  1.15501241e+00 -1.05087975e+00  3.39156164e-01\n",
      "  3.94128443e-01  5.74875300e-01 -1.54682742e+00  1.81959222e+00\n",
      "  8.22481525e-01  3.36282205e-01 -3.27201230e-01  3.27784857e-02\n",
      " -4.38179017e-01  1.04807379e-01 -1.08877507e+00 -4.19600562e-01\n",
      " -5.69231018e-01 -1.21510854e+00 -7.33465134e-01  5.19588057e-01\n",
      " -2.05550028e+00  4.41345637e-01 -1.59781060e+00  1.73104724e+00\n",
      " -1.30725600e-01 -2.96749885e-01 -1.13522560e-02 -2.19808866e+00\n",
      " -1.30633787e+00  1.15372428e+00  1.80070706e-01 -4.68020353e-01\n",
      " -1.55800073e-01 -3.96439839e-01 -8.07413678e-02 -4.85364798e-01\n",
      " -7.12689081e-01  8.52632976e-02 -4.14950296e-02 -9.69516100e-01\n",
      " -2.34696313e-01  5.09656978e-01 -3.68069781e-01  1.46724490e+00\n",
      "  4.47559584e-01  8.76792407e-01 -5.89377937e-01 -2.91304857e+00\n",
      " -3.22363269e-01 -8.74472989e-01  1.27247633e-01 -1.61516950e+00\n",
      "  9.32987861e-02  1.90120614e-01 -4.50170967e-01 -8.17918770e-01\n",
      " -1.63479821e+00  1.56074293e+00  1.37068552e+00  1.51773933e+00\n",
      "  8.74834163e-01  1.98620834e+00  2.05723867e+00 -7.97599535e-01\n",
      "  6.52278270e-02  2.40172573e+00  3.59105264e-01  1.00627775e+00\n",
      " -7.97184256e-01 -1.55477414e+00 -2.17987034e+00  1.45243139e-01\n",
      " -1.39698794e+00  6.28532968e-01  1.27478256e-01  7.43327281e-01\n",
      "  5.46446353e-01  7.49090932e-01 -9.85549815e-02  1.17819069e-01\n",
      " -1.12430178e+00  1.47814539e+00  1.54363869e-01  8.46174639e-01\n",
      " -6.55357565e-01  2.15733133e+00 -2.34995891e-01 -4.12733871e+00\n",
      "  1.46915162e+00  1.55883514e+00 -4.05520946e-01  4.66498978e-01\n",
      "  3.47726849e+00  7.56944709e-03  5.58047533e-01 -2.34186749e+00\n",
      "  1.03858847e+00 -1.11572770e+00 -6.00740998e-01  1.73271146e+00\n",
      "  6.94065297e-01  3.89337289e-02  2.06476763e-01 -3.83419185e-01\n",
      "  1.88072439e+00  6.25773752e-01 -2.88523504e-01 -1.93946578e-01\n",
      "  1.15616772e+00 -3.29201736e-01  6.46235493e-01 -8.86064997e-01\n",
      " -3.89711388e-01 -3.62621591e-01  2.94623525e-01 -5.14641489e-01\n",
      "  9.54307377e-01  2.67307697e-01 -2.91803884e-01 -1.48916109e+00\n",
      "  2.67577564e-01 -5.65003844e-01  6.22803393e-01  1.12462000e-01\n",
      " -8.06756077e-01  4.15778485e-01 -8.87865967e-02  9.80784759e-01\n",
      "  3.32341361e-01 -8.30945915e-01  2.78679547e-01 -2.28988870e+00\n",
      "  1.15685671e+00  4.35391800e-01  3.22756884e-01  1.89209755e-01\n",
      "  8.73852898e-01 -2.94486923e-01  4.17647626e-01  8.42793676e-01\n",
      " -6.10748427e-01 -5.95730644e-01 -3.18092255e-01  1.99317232e-01\n",
      "  9.27790678e-01  2.92823457e-01 -5.00054825e-01  5.36244582e-01\n",
      " -6.71707015e-01  4.13939391e-01 -4.85718076e-01 -3.58958379e-01\n",
      "  1.69507542e-01 -5.68872186e-01  9.18126640e-01 -1.04108648e+00\n",
      " -1.40419643e+00 -5.28058061e-01 -3.66856201e-01 -1.06953030e-02\n",
      "  1.44863608e+00 -6.67610617e-01 -8.00639734e-01 -1.69474820e+00\n",
      "  1.45693395e+00  1.54315948e+00  1.73483911e-01  1.06198647e+00\n",
      " -4.37342406e-01 -3.13050680e-01  1.28248809e+00  4.50915008e-01\n",
      " -7.94446317e-01  1.12892253e+00 -3.78751084e-01 -7.99216575e-01\n",
      "  2.03297472e-01 -9.62752022e-01  7.62017456e-01  4.19180798e-01\n",
      "  1.68807674e-01 -1.39642462e+00  2.25138799e-01  1.99694839e-01\n",
      "  9.82054911e-02 -7.08726388e-01  1.47273355e+00  7.49469875e-01\n",
      "  3.06055840e-01 -6.20423590e-01 -3.99529763e-01 -3.24594585e+00\n",
      " -1.10197401e+00  1.85062587e-01 -5.64130701e-01 -9.69614926e-01\n",
      " -4.87302255e-01 -1.37825112e+00  1.56761273e-01 -1.89756676e-01\n",
      "  7.20706928e-01 -2.51313570e-01 -1.19506124e+00  3.18076697e+00\n",
      " -2.17909531e+00 -2.29135183e+00 -8.26402335e-01 -9.42426912e-02\n",
      " -1.17649354e+00  2.59525967e-01  9.96226210e-01  3.12296525e-01\n",
      " -6.63042855e-01  1.24993948e+00  9.97869373e-01  6.15477285e-01\n",
      "  5.61806521e-01 -1.13527760e-01 -5.29184122e-02 -1.31791145e+00\n",
      "  3.70536526e-01 -1.55413089e-01 -3.62233638e-01  7.27893397e-01\n",
      "  1.23439324e-01  7.79962550e-01 -2.65435403e+00  1.61865217e-01\n",
      "  5.98560580e-01  1.23831921e+00  6.29531238e-01  1.54851457e+00\n",
      "  3.03265134e-01  2.26212236e-01 -4.23512135e-01 -9.51648659e-01\n",
      " -2.37425124e+00 -1.48866582e-01 -1.82235007e-02 -8.70092666e-01\n",
      " -1.10299427e-01 -1.19957236e+00  7.28085038e-01  2.05697904e-01\n",
      "  8.95503567e-01  5.34714129e-01 -5.06262496e-01 -3.64831909e-01\n",
      "  8.75881718e-01  1.63405216e+00  1.06484150e+00 -4.69142346e-02\n",
      " -7.42547099e-01  5.49831418e-01  4.81143920e-01  1.22052693e+00\n",
      "  1.18317166e+00  1.04377454e+00  4.58741246e-01  1.07551727e+00\n",
      " -2.90034060e+00 -6.35424364e-01 -1.29818980e+00  2.97414690e+00\n",
      "  1.68410042e+00 -1.23406081e+00 -2.47656640e-01 -1.82506753e+00\n",
      "  1.11290656e+00  2.19538294e+00  3.88883528e-01  3.09571954e-01\n",
      "  1.85862465e-01 -8.99194080e-01 -2.37764530e-01 -5.45078205e-01\n",
      "  6.61491093e-01 -1.68048541e+00 -4.95149054e-01 -4.91979159e-01\n",
      "  8.33239901e-01  2.30795468e+00  2.67428693e-01 -8.31088094e-01\n",
      " -5.48090756e-01  1.85579474e+00 -8.36426316e-01 -3.30018731e-01\n",
      "  2.63611610e+00  4.62440896e-01 -9.46693922e-01  1.15216609e+00\n",
      " -3.64594187e-01  1.02876715e+00 -1.60426998e+00  4.85297761e-01\n",
      "  1.59345622e-02 -6.22887680e-02  1.33364673e-01  3.49977620e-01\n",
      "  1.84366100e+00 -7.27543176e-01 -1.24604893e-01  1.47096889e-01\n",
      "  1.99991504e+00 -1.10682540e+00 -7.27420420e-01 -1.32344430e+00\n",
      "  2.07561415e+00 -1.29792909e+00 -1.36888925e-01 -3.37215908e-02\n",
      " -9.92190853e-01 -9.27402271e-01  1.26557256e+00  1.36792641e+00\n",
      "  1.65224994e+00 -3.36816424e-01 -4.14300892e-01  9.15798565e-01\n",
      "  1.97128812e+00 -1.94259800e-01  2.86371099e-01  5.18149461e-01\n",
      " -1.02650800e+00  1.01680152e+00 -2.21526992e+00 -1.10851269e+00\n",
      " -7.82824551e-01  5.50926041e-01 -2.54672773e-01 -3.74103042e-01\n",
      " -7.91622098e-01 -1.64359102e+00 -5.10432377e-01 -5.09419396e-03\n",
      "  1.78139208e-01 -4.97975037e-01  1.11979387e+00  4.54532185e-01\n",
      " -7.68317047e-01  4.21998217e-01 -2.56429319e+00 -7.57379848e-01\n",
      "  2.15624746e+00  9.39951890e-01 -1.34576524e+00  2.40403085e+00\n",
      "  8.52618850e-01  7.14479051e-01  3.12366432e-01  2.39152420e-01\n",
      " -8.92630600e-01 -1.48236666e+00 -8.99367666e-01  2.21018386e+00\n",
      " -8.89636071e-01  6.53894427e-01  1.93805393e+00 -2.74747032e+00\n",
      "  1.28629484e-01 -4.41108398e-01 -7.66222194e-01 -3.85595192e-01\n",
      "  2.45442092e-02  3.93113988e+00  1.00603753e+00 -1.02323239e-01\n",
      " -1.99357531e-01  1.53989239e+00  8.55977116e-01  1.14589710e+00\n",
      "  4.43957399e-01  4.21765763e-01 -1.75091607e-01 -2.26285973e+00\n",
      " -9.73307815e-01  4.45461325e-01 -1.41648163e-01  4.93152897e-01\n",
      "  8.35063530e-01 -1.53207801e+00  2.66195116e-01  3.32902756e-01\n",
      " -2.61688973e-01  1.27142949e+00  2.79095481e+00  1.05354932e+00\n",
      "  1.39351775e+00 -1.27503277e+00  2.36107110e-01  1.17207916e+00\n",
      "  1.15505891e-02  1.72387755e+00  2.89127447e-01 -4.22642543e-02\n",
      " -3.98243926e-01  2.08421701e+00  1.20682463e+00 -5.02913565e-01\n",
      "  7.89122101e-01  6.04012972e-02  8.70606526e-01 -4.09879912e-01\n",
      "  5.70616068e-01  2.03833905e+00  3.24882106e-02 -1.92964394e+00\n",
      " -3.14331872e-01 -1.07464904e+00  4.72327614e-01 -7.48093633e-01\n",
      "  3.15953035e-01  1.28510572e+00 -8.15998712e-01 -1.04647382e+00\n",
      " -1.17186870e+00 -5.28301473e-01 -8.72271029e-01  7.22006833e-01\n",
      "  1.06431039e+00 -4.27476668e-01  1.85841826e+00  7.69094417e-01\n",
      "  2.35894635e-01  6.58817751e-01 -1.57165667e-01  2.99301549e-01\n",
      " -1.36484652e+00  1.32823589e+00  5.02501656e-01 -1.89884839e+00\n",
      "  6.28430004e-01  1.42439088e-01  5.39125632e-01 -2.13931582e+00\n",
      " -9.63656871e-01 -2.57127447e-01  2.24602990e-01 -1.57990456e+00\n",
      " -1.61401308e-01  3.08269224e-01 -1.26449952e+00 -8.79089770e-01\n",
      " -9.45965578e-01 -1.20551057e+00 -3.06915712e-01 -4.47926003e-01\n",
      "  2.05388792e+00  3.08806893e-01  7.94055645e-02  8.62638708e-01\n",
      " -1.62918788e-01  5.38312027e-03  1.87734189e+00  3.17031753e-01\n",
      " -1.68597611e+00 -1.83757452e-01  3.12076810e-01 -6.96063996e-01\n",
      " -1.87350570e+00 -6.29869758e-01  1.11291401e-01  4.56295169e-01\n",
      "  1.64605027e-02 -2.33891447e-01  1.09195603e+00  3.37752416e-01\n",
      "  1.71654320e-01  1.49942322e-01  1.13695162e+00 -3.87470387e-01\n",
      " -2.97044473e-01 -1.58693464e+00  1.05887129e+00 -7.35611459e-01\n",
      " -1.78396230e+00 -5.11073968e-01  1.56062362e+00 -1.19497335e+00\n",
      "  9.85718807e-01  2.59761823e-01 -6.69258296e-01 -9.25455123e-01\n",
      "  4.54012299e-01  1.17542419e-01 -1.09557135e+00  2.46517845e+00\n",
      "  3.47481725e-01 -3.51604643e-01  2.81758993e-01  1.90384268e+00\n",
      " -1.75767506e-01  9.95424342e-01 -6.14514235e-03  3.06084518e+00\n",
      "  8.27003929e-01  9.03253700e-01  2.78430499e+00  2.25316295e-01\n",
      " -1.58387827e+00 -5.10074170e-02 -4.60794657e-01  2.12959493e+00\n",
      " -2.38950070e-01  3.26999951e+00 -4.00744283e-01  1.89791423e-01\n",
      "  3.53154294e-01  1.30100923e+00 -6.35187754e-02  2.05394696e+00\n",
      "  1.19051058e+00 -1.20594406e+00 -9.50953386e-01 -2.11915448e+00\n",
      "  5.27119722e-01 -1.35392167e+00 -4.70106049e-01  9.25481547e-01\n",
      " -5.23921998e-01  1.55520746e+00 -7.55689083e-01  1.56919018e-01\n",
      "  5.53760028e-01  7.77563528e-01 -3.15262941e-01  2.17550994e-01\n",
      " -1.06581317e+00  1.03265133e+00 -1.93912371e-01 -9.44033022e-01\n",
      "  2.75103082e-01 -9.27821811e-01  2.08846610e-02  3.98291883e-01\n",
      "  2.19479473e+00 -1.10194940e+00  2.20933091e+00  4.03231337e-01\n",
      " -1.15362389e+00  2.48815775e-01  4.35911779e-01 -5.82020735e-01\n",
      "  1.21140303e-01 -6.25831553e-01  9.90519910e-01  4.41388450e-01\n",
      "  7.67926489e-01 -3.79234524e-01  5.21663250e-01  4.60028289e-01\n",
      " -6.25434164e-01 -1.41747684e+00  7.59831260e-01 -6.16746541e-01\n",
      " -1.75873440e+00 -7.37123129e-02  7.29431575e-01 -2.75459276e+00\n",
      "  1.08793052e-01  8.58616851e-02  1.72218057e+00  1.34730265e+00\n",
      " -9.55201741e-01  2.88452494e-01  8.20501729e-01 -3.93165712e-01\n",
      " -3.86734315e-01 -1.01143832e+00  9.34681099e-01  1.50820605e+00\n",
      " -5.42221896e-01 -1.34144927e+00 -7.04173316e-01  2.06857857e+00\n",
      " -1.91415407e+00  8.60794538e-01 -6.45292612e-01 -4.73117585e-01\n",
      "  5.21602901e-01 -1.14737860e+00 -6.74548242e-01 -1.01027839e+00\n",
      " -5.47631139e-01  2.62894780e-01  8.79877299e-01 -2.94359823e+00\n",
      " -1.35020511e+00  1.46439443e+00 -1.09974789e-01 -3.23026051e-01\n",
      " -1.10069921e+00 -6.62000319e-01 -4.99308822e-01 -4.15648889e-01\n",
      "  1.10794590e-01  7.05554302e-01 -3.53310493e+00  1.07314062e+00\n",
      " -9.90757761e-01 -1.34048083e-01  1.27557809e+00 -1.44210683e-01\n",
      " -1.91703620e+00  9.58356743e-01 -7.28342443e-01  2.75785817e-01\n",
      " -5.96993019e-01  1.37162156e+00 -1.26096737e-01 -1.94539885e+00\n",
      " -5.67585793e-01 -7.90630255e-01  2.97847294e-01  7.91680802e-01\n",
      "  1.34262796e+00  9.44072502e-01 -1.05350408e+00  5.65520452e-01\n",
      " -3.12671176e-02 -1.19850266e+00 -4.03088476e-01  1.18213108e+00\n",
      " -7.66229371e-01  1.69750849e+00 -1.24434055e+00 -9.60992051e-01\n",
      "  2.21961949e-01 -3.89856288e-01 -1.35166789e-01 -1.05655058e+00\n",
      "  8.64948652e-01  4.15673911e-01 -7.49931854e-01  1.10425089e+00\n",
      " -6.85678782e-01  9.38577652e-01  1.04748047e+00 -2.42177476e-01\n",
      " -4.44561048e-01  1.23526798e+00  9.62984266e-01  5.93781132e-01\n",
      " -1.05546223e+00 -1.26801676e+00  9.60478653e-01  2.06383397e-01\n",
      "  9.58565757e-01 -4.66699855e-01  1.31354502e-02  6.66244923e-01\n",
      " -3.22782661e-01 -8.04760424e-01  1.24081505e-01  1.03669370e+00\n",
      "  1.66032193e+00  1.68113185e-01  2.01881889e+00  1.24839623e-01\n",
      "  1.25002295e-01  1.92037955e+00  3.33028587e-01  1.08297475e-01\n",
      "  4.61428193e-01 -6.26423893e-01 -1.67711443e+00 -3.54786817e-01\n",
      "  1.16623246e+00  1.14556743e-01 -2.85550759e-02  2.45491251e-01\n",
      "  1.11979211e+00  9.72645082e-01  1.09898714e-01  1.72150906e+00\n",
      "  2.20069308e-01  3.25638663e-01  1.42207596e+00  3.68553582e-01\n",
      "  8.73217653e-01 -3.95890516e-01  6.73194118e-01 -5.01590379e-01\n",
      " -3.65481677e-01  2.33784046e+00 -6.35485058e-01 -9.05629758e-01\n",
      "  1.87382750e+00  1.12450130e+00 -1.72344232e-01  5.13770295e-01\n",
      "  1.36665255e+00 -9.43788741e-01  7.88463066e-01 -9.89679028e-01\n",
      " -1.68473444e+00  9.17029290e-01  3.07858546e-01 -1.37002425e+00\n",
      "  3.45808618e-01 -5.49607559e-01 -7.58498991e-01  1.97739152e+00\n",
      " -1.62774009e+00  1.03569964e+00 -5.23407794e-01 -2.56156833e+00\n",
      " -7.41171260e-01 -8.66327472e-01  3.91767815e-01 -7.99087264e-01\n",
      "  1.55076681e-01  4.85216996e-01 -6.47753081e-01  8.27519871e-02\n",
      " -6.48737477e-01  2.44661227e+00  1.96312958e+00  4.39761578e-01\n",
      "  1.38812521e+00 -2.17596830e-01 -5.07878371e-01  2.60278646e-01\n",
      " -9.60053117e-01 -9.45650460e-01 -6.73712606e-01  1.59234486e+00\n",
      " -1.02936655e-01 -6.77766424e-01  1.06427646e+00 -1.23932437e+00\n",
      "  5.86009717e-01 -6.14350773e-01 -1.96438235e+00 -3.37931657e-01\n",
      " -1.58410150e+00 -2.11696972e+00 -1.56103639e-01 -1.02068603e+00\n",
      " -1.37620480e+00 -8.39363103e-01 -6.81997333e-01 -2.50561759e+00\n",
      "  9.24591962e-01 -1.60832699e-01  1.59218933e+00  6.93622206e-01\n",
      "  1.64739915e+00 -1.17622860e-01  5.01568317e-02  9.68201214e-01\n",
      " -9.04248727e-01 -1.06491702e+00  5.75397790e-01 -1.16358063e+00\n",
      "  1.10733864e-01 -8.30529488e-01  1.32644252e-01  3.37273748e+00\n",
      "  9.61773224e-01 -1.17001804e+00 -1.36602038e+00 -1.66757643e-01\n",
      "  1.00966103e+00 -6.80176993e-01 -2.15782346e-01  4.44366530e-01\n",
      "  1.17156598e+00  2.15789991e-01 -1.57452541e+00 -2.10903230e+00\n",
      "  1.38205586e+00  7.02628386e-01 -1.02074285e+00  9.71672153e-01\n",
      "  3.43901201e-01 -2.95883708e-01  2.05007563e+00  1.36391951e+00]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([  3.,  20.,  90., 278., 338., 198.,  58.,  13.,   1.,   1.]),\n",
       " array([-4.12733871, -3.16427526, -2.20121181, -1.23814837, -0.27508492,\n",
       "         0.68797852,  1.65104197,  2.61410542,  3.57716886,  4.54023231,\n",
       "         5.50329575]),\n",
       " <BarContainer object of 10 artists>)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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nJwcffTRR9EcBwAAGCJq3yS7fft2lZaWav369frxj3+s1157TUVFRfr00081ePDgaI0lqXd+syIAALEkamdQVq9ereLiYj311FO6//77tXbtWvl8PlVVVUVrJAAAYIionEHp7OxUY2OjfvOb34TtLywsVH19/RXrQ6GQQqGQfTsQCEiSgsFgj8z3f/q/HnlcAMDNK3eUR3uE20J5oPtf529+b1uW9Z1roxIo586dU1dXl9xud9h+t9stv99/xfrKykotX37l2y4+n6/HZgQA4Hb2ouvFHnvstrY2uVyu666J6l8zdjgcYbcty7pinySVl5errKzMvv3111/r3//+t1JSUq66vicEg0H5fD41NTUpKSnplvzM2x2veXTwukcHr/utx2t+61mWpba2Nnm93u9cG5VASU1NVVxc3BVnS1paWq44qyJJTqdTTqczbN8PfvCDnhzxmpKSkvgf+RbjNY8OXvfo4HW/9XjNb63vOnPyjahcJJuQkKCcnBzV1taG7a+trVVeXl40RgIAAAaJ2ls8ZWVlmjVrlkaMGKGRI0fq9ddf1+nTpzVv3rxojQQAAAwRtUCZMWOGzp8/r+eff17Nzc3Kzs7Wrl27lJGREa2RrsvpdGrZsmVXvNWEnsNrHh287tHB637r8ZqbzWF9n8/6AAAA3EL8LR4AAGAcAgUAABiHQAEAAMYhUAAAgHEIFAAAYBwC5SaEQiE9+OCDcjgcOnz4cLTHiWmnTp1ScXGxMjMz1bdvX919991atmyZOjs7oz1azFm/fr0yMzN15513KicnRx999FG0R4pZlZWVevjhh5WYmKi0tDRNnjxZn332WbTHuu1UVlbK4XCotLQ02qPgWwiUm7BkyZLv9fcEcPP+/ve/6+uvv9Zrr72mY8eOac2aNfr973+vpUuXRnu0mLJ9+3aVlpbqueee08cff6yf/OQnKioq0unTp6M9Wkyqq6vTggULdPDgQdXW1uqrr75SYWGhLl68GO3RbhsNDQ16/fXXNXz48GiPgsvwPSg36IMPPlBZWZnefvttDR06VB9//LEefPDBaI91W3n55ZdVVVWlzz//PNqjxIzc3Fw99NBDqqqqsvfdf//9mjx5siorK6M42e3hyy+/VFpamurq6vTYY49Fe5yY197eroceekjr16/XCy+8oAcffFBr166N9lj4H86g3IAvvvhCc+fO1ZYtW9SvX79oj3PbCgQCSk5OjvYYMaOzs1ONjY0qLCwM219YWKj6+vooTXV7CQQCksT/17fIggUL9NOf/lQFBQXRHgVXEbWvuu+tLMvSnDlzNG/ePI0YMUKnTp2K9ki3pX/84x969dVXtWrVqmiPEjPOnTunrq6uK/6iuNvtvuIvj6P7WZalsrIyPfroo8rOzo72ODGvurpaf/3rX9XQ0BDtUXANnEH5n4qKCjkcjutuhw4d0quvvqpgMKjy8vJojxwTvu/r/m1nz57VhAkT9LOf/UxPPfVUlCaPXQ6HI+y2ZVlX7EP3W7hwoT755BO99dZb0R4l5jU1NemZZ57R1q1bdeedd0Z7HFwD16D8z7lz53Tu3LnrrvnhD3+oJ598Uu+//37YP9hdXV2Ki4vTL37xC23atKmnR40p3/d1/+YfkbNnz2r06NHKzc3Vxo0bdccdNHZ36ezsVL9+/fTHP/5RU6ZMsfc/88wzOnz4sOrq6qI4XWwrKSnRjh07tH//fmVmZkZ7nJi3Y8cOTZkyRXFxcfa+rq4uORwO3XHHHQqFQmHHEB0ESoROnz6tYDBo3z579qzGjx+vP/3pT8rNzdWgQYOiOF1s+9e//qXRo0crJydHW7du5R+QHpCbm6ucnBytX7/e3jdkyBA98cQTXCTbAyzLUklJiWpqarRv3z5lZWVFe6TbQltbm/75z3+G7fvlL3+pH/3oR3r22Wd5i80QXIMSocGDB4fdvuuuuyRJd999N3HSg86ePav8/HwNHjxYv/vd7/Tll1/axzweTxQniy1lZWWaNWuWRowYoZEjR+r111/X6dOnNW/evGiPFpMWLFigbdu26d1331ViYqJ9rY/L5VLfvn2jPF3sSkxMvCJC+vfvr5SUFOLEIAQKeoU9e/boxIkTOnHixBUhyEnA7jNjxgydP39ezz//vJqbm5Wdna1du3YpIyMj2qPFpG8+zp2fnx+2f8OGDZozZ86tHwgwCG/xAAAA43CFIQAAMA6BAgAAjEOgAAAA4xAoAADAOAQKAAAwDoECAACMQ6AAAADjECgAAMA4BAoAADAOgQIAAIxDoAAAAOP8P/PpjEsxUP11AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## T Distribution using scipy stats\n",
    "from scipy.stats import t\n",
    "#T_Dist = t.rvs(df = 10,  size=1000) #df - no. of independent observations in a set of data, degree of freedom\n",
    "T_D = np.random.standard_t(10, size=1000) ##1st argument is dof, 2nd is sample size\n",
    "print(\"samples from T Distribution using scipy stats library are: \", T_D)\n",
    "pyplot.hist(T_D,color='purple')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([89.,  8.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  2.]),\n",
       " array([9.36130270e-03, 5.83245097e+00, 1.16555406e+01, 1.74786303e+01,\n",
       "        2.33017200e+01, 2.91248096e+01, 3.49478993e+01, 4.07709890e+01,\n",
       "        4.65940787e+01, 5.24171683e+01, 5.82402580e+01]),\n",
       " <BarContainer object of 10 artists>)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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vlr6w7B9NrtevBh2wFm5cmUaPHhw+vWvf51eeuml1NTUlI444oj0j3/8o7+3dsDo7u5O69atS+vWrUsRkRYsWJDWrVuXXn/99ZRSSnfeeWeqrq5ODz/8cNqwYUO66qqr0ogRI1JXV1c/77z/fO9730vV1dXpqaeeSm+99Vbp9t5775XOMbdys2bNSk8//XTavHlzevHFF9OPfvSjdMghh6RVq1allMzr4/rvd7ukZG578oMf/CA99dRTadOmTenZZ59NX/3qV1NVVVXp+76Z7dlzzz2XBg0alO6444702muvpd/85jdp6NCh6cEHHyyd0xez+1TER0op/fKXv0zHH398Ouyww9JZZ51VejskH3ryySdTRPS4XXfddSmlD99eNWfOnFRbW5uKxWI6//zz04YNG/p30/1sT/OKiLRs2bLSOeZW7vrrry/9f/jZz342XXDBBaXwSMm8Pq7/jQ9z6+mKK65II0aMSIMHD051dXVpypQpaePGjaXjZrZ3f/zjH9OoUaNSsVhMJ510UlqyZEnZ8b6YXSGllHp9fQYAoEIH/Ws+AIADi/gAALISHwBAVuIDAMhKfAAAWYkPACAr8QEAZCU+AICsxAcAkJX4AACyEh8AQFbiAwDI6v8BwdeinSPoNFoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##F distribution using scipy library\n",
    "from scipy.stats import f\n",
    "#F_Dist = f.rvs(dfn=10, dfd=20,loc=0, scale=1, size=1000) # dfn, dof for numertaor\n",
    "## F Distribution  ---numpy.random.f(dfnum, dfden, size=None), dfnum = dof in numerator\n",
    "#Draw samples from an F distribution.\n",
    "F_Dist = np.random.f(dfnum = 3, dfden = 2, size=100) # using numpy library\n",
    "#print(\"samples from F Distribution using scipy stats library are: \", F_Dist)\n",
    "pyplot.hist(F_Dist,color='purple')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "samples from Gamma Distribution are:  [1.30830744 0.85817535 2.67651437 0.47099064 0.59663439 2.72002652\n",
      " 0.07835516 4.19807286 1.33748819 1.06603289 1.01186253 3.30058904\n",
      " 3.14092088 0.35080114 4.67318341 0.7225863  2.67868022 0.96866806\n",
      " 2.50917902 0.85312433 1.44821183 2.36584215 0.92475485 2.94133247\n",
      " 3.92391631 1.28057683 3.25009493 0.80514725 2.65218143 0.43341009\n",
      " 1.03939756 1.1795815  1.08969086 1.64273227 1.21681621 2.70414891\n",
      " 1.03458219 0.40226905 5.54624008 1.57818338 3.4391358  3.72246731\n",
      " 2.01807995 2.5421433  1.07208638 1.34039238 1.23652934 1.51724257\n",
      " 0.83370614 1.47665875 1.57141726 1.67814585 0.48887251 0.8478779\n",
      " 3.5527041  0.59357498 2.48703377 2.23840648 0.56059697 0.24694256\n",
      " 1.13107796 0.88862864 0.88069399 1.49801019 0.79652939 2.1217022\n",
      " 0.19286976 5.12496915 1.55005909 1.51160834 3.68876121 0.46213622\n",
      " 3.05290802 1.31963925 2.88871414 1.73753036 1.77424758 1.07614848\n",
      " 0.4093067  3.39747005 1.74552088 2.79826166 1.19404191 0.7815277\n",
      " 3.65592806 2.09007032 0.99216037 1.0219614  2.90213374 1.15941112\n",
      " 0.72835261 1.61250575 1.74107156 1.19944375 0.47800376 1.5207025\n",
      " 3.07240608 2.85898773 3.92605249 0.46509563 2.46798928 1.54836615\n",
      " 1.24855024 0.39609198 2.66074292 1.36271003 2.52170276 0.30999519\n",
      " 1.4554072  2.15841491 2.22420891 3.34481072 0.98697163 2.40991416\n",
      " 4.11578664 5.61161315 4.97415358 0.73686092 1.30038752 2.15379258\n",
      " 0.85588353 1.83724636 3.13121755 1.9447159  1.13925062 0.54287806\n",
      " 3.69125674 0.33957934 1.36320901 2.6615443  2.79451754 1.10892318\n",
      " 2.40359643 4.99225248 0.80570618 1.3131556  5.68323537 1.49093526\n",
      " 0.27950152 3.61855634 4.03178992 0.50044849 1.52245883 3.1638578\n",
      " 0.79031907 2.78082554 0.67916096 1.16846748 0.87943395 1.43294779\n",
      " 1.68945093 2.01243658 1.14416264 1.22206901 1.04501571 3.18566532\n",
      " 9.25918038 1.4560252  3.47913516 0.91615653 0.72145925 1.98902712\n",
      " 0.95207793 3.38428221 2.91836741 1.59208266 2.87775497 2.73195984\n",
      " 3.91207543 1.2942524  0.04917766 3.46834861 0.20915597 0.58659606\n",
      " 3.45808776 3.32258387 1.28108417 0.9630813  2.5552895  0.25070355\n",
      " 0.74844623 0.77972022 0.93813708 0.46883589 1.9607517  1.01477645\n",
      " 1.58992473 2.32719139 2.52251633 1.43870701 1.69498489 1.56245064\n",
      " 1.82004753 3.32179341 1.50058266 1.02315416 0.21871252 1.11699394\n",
      " 1.69813736 0.17896299 2.68056905 3.41364592 2.7861665  4.66353359\n",
      " 1.54070587 1.67518515 3.88963007 1.48929077 0.9391686  1.91439411\n",
      " 0.51582942 2.38387237 1.17311403 0.5803662  2.00728882 0.37238784\n",
      " 2.50077565 4.59566311 4.34209193 0.51103051 1.03022504 1.42899611\n",
      " 1.68221525 1.82800402 0.74316283 1.90870185 1.36303157 3.6706991\n",
      " 2.30999814 1.24697282 0.37375634 2.28202697 0.82665867 0.93199077\n",
      " 0.7632056  2.703313   1.96458703 1.95510961 2.81869592 0.50502772\n",
      " 2.36593228 1.53567054 3.17155744 4.3237874  1.52158983 1.11122177\n",
      " 0.94396017 1.37464227 4.03611241 1.99522883 2.95632675 2.27517518\n",
      " 0.44364767 0.19085253 1.19120866 1.81119646 2.1155775  1.67154315\n",
      " 0.28630317 2.98387902 0.43682601 2.16118351 2.0234772  4.65892868\n",
      " 2.14531539 2.56825943 0.13766816 1.76358358 1.40131295 4.34426723\n",
      " 1.2020907  1.60945985 0.25342168 3.50653822 1.84448399 3.58501409\n",
      " 1.22805155 2.67495091 0.58096127 2.09251727 2.09162209 1.24590844\n",
      " 0.75111816 0.99657423 2.02037071 3.28631679 0.98479236 1.8656782\n",
      " 0.56755457 0.09926708 1.97754602 2.98499797 0.32865    0.53482657\n",
      " 1.79137338 2.4882464  1.77281101 1.73413111 3.91448304 2.60473463\n",
      " 1.16361977 0.9324305  0.94294916 2.0257429  3.15746052 1.57660762\n",
      " 3.111075   1.12083274 3.75966507 4.89299295 3.28604824 3.95871728\n",
      " 1.79338551 1.93963292 1.91910638 0.66076727 1.26460764 1.40290004\n",
      " 2.97161037 1.32887713 2.55136685 2.24003964 2.28689372 1.95089003\n",
      " 0.76159967 5.58482641 0.59321852 0.6230504  0.614237   0.46823652\n",
      " 0.8991974  1.76622694 0.06731298 0.92988262 0.37783853 8.05824665\n",
      " 1.16422981 2.69596995 0.42632505 1.88218221 2.36084791 1.88687406\n",
      " 4.38796533 1.53839423 4.04959897 2.06683398 1.46952442 0.90388613\n",
      " 0.88425229 1.28025908 4.33895977 2.62337982 1.04860767 2.95181603\n",
      " 2.75194625 1.29088261 1.92451582 0.74974466 2.68735025 6.65903434\n",
      " 2.83595215 3.16247075 2.10870116 1.74522432 1.43449414 3.32034257\n",
      " 1.91240927 3.92929025 5.20552106 1.60418544 4.87102372 3.59883625\n",
      " 1.5659896  0.34381738 1.57592354 1.63595619 1.34531286 1.0590412\n",
      " 1.03488234 1.19768002 1.05276099 1.15246286 2.66309868 0.86612906\n",
      " 3.79578387 1.25111745 3.37255477 2.62496606 3.04584593 1.12460831\n",
      " 4.21033478 0.43721234 0.84626418 2.65866471 0.59750893 1.09403662\n",
      " 4.28879102 4.26160731 3.25200433 0.94701486 2.48703915 0.94059667\n",
      " 1.76706047 0.13227263 0.1749101  1.42883438 0.8255601  1.56888894\n",
      " 0.82381726 1.59995635 1.25916363 2.39933621 2.33464061 2.04411567\n",
      " 1.31949761 3.6341495  0.41634114 1.50098774 3.90443167 2.55716661\n",
      " 3.19654981 1.00658741 2.31303231 6.12104326 2.7269011  1.14655726\n",
      " 2.24704754 2.84796028 4.58146619 3.15935301 1.06365385 3.64481221\n",
      " 1.25177191 5.58112312 1.68570974 1.6439668  1.08925063 1.34797582\n",
      " 0.54072519 1.5504597  1.91889416 4.26311618 0.82111939 0.90337378\n",
      " 2.97643818 0.63770729 2.45457427 2.36015151 5.55879148 1.19402135\n",
      " 3.50766725 1.29262266 6.91721655 1.86141411 1.7059131  1.19323006\n",
      " 2.61863122 1.27667833 1.28575546 0.89419833 2.3323405  0.86625918\n",
      " 1.27029342 2.00514756 3.60841167 0.72118984 2.1002703  1.32551543\n",
      " 0.65910259 2.03507071 4.80643428 5.57025968 1.13848127 0.38712619\n",
      " 2.35503656 3.47910318 1.65775044 1.4124995  2.83528165 1.15398727\n",
      " 2.07925535 1.01049441 3.64269853 0.33356407 1.01731663 3.36646428\n",
      " 2.28422469 5.90687768 1.73859162 0.05848051 3.73587492 0.68790748\n",
      " 3.33076173 1.40854118 2.02777721 1.67747296 2.61852962 5.29505824\n",
      " 1.97885286 1.10470151 2.15068826 2.81508979 1.72271374 3.24809902\n",
      " 3.85784383 3.0651871  1.71091194 0.08616855 2.94293294 1.07179998\n",
      " 3.78730241 1.96772043 2.68159862 2.0108768  0.79309679 1.27834289\n",
      " 0.8315656  1.10953285 2.63135819 4.40158165 1.08415965 0.65704273\n",
      " 0.40948241 4.0011253  1.805213   3.29161285 2.94408822 1.03972128\n",
      " 1.09107215 1.18088189 3.07337571 3.91711571 4.80124995 0.58173736\n",
      " 4.0482068  1.77441204 0.4921743  1.11865976 0.26276982 1.28716234\n",
      " 1.55829383 3.89534736 3.70211005 0.4347239  0.88607699 1.47666361\n",
      " 0.92939393 0.39347493 1.25982531 2.94590843 1.05590054 2.28215993\n",
      " 1.00370708 0.64455285 1.69088292 1.84725542 2.96010288 6.56968109\n",
      " 2.32105759 2.05721253 3.25011005 0.49962633 0.7741391  1.67386425\n",
      " 4.54946291 0.83931775 1.59491137 1.02978635 0.3655169  5.78248039\n",
      " 0.65347801 2.3249013  1.6028012  0.31199579 1.47464567 5.11191708\n",
      " 2.0188687  0.21996127 0.68331006 4.8366681  0.22047833 2.29959819\n",
      " 0.77098423 0.98202614 5.02301726 1.85195036 2.08639303 1.98608496\n",
      " 1.06727709 2.41372259 2.11669132 1.30028829 5.96568879 0.48161847\n",
      " 0.61873782 1.22123926 0.66336088 1.66167365 1.70360231 1.59963026\n",
      " 0.54062171 0.897611   1.87798849 0.99625327 0.57247833 2.57830356\n",
      " 2.01805257 3.2984377  1.572022   0.54685471 2.06641906 0.63470949\n",
      " 4.59839103 5.87694328 2.76178783 1.5890849  3.68083852 4.14713596\n",
      " 3.00220539 3.17563989 1.70040555 2.69147685 3.18862826 0.17563305\n",
      " 0.2020896  1.84350684 0.52759666 0.70353147 3.34619445 1.86115007\n",
      " 2.77329168 2.02189922 3.08092046 2.59390172 0.70526117 2.12143883\n",
      " 2.62276225 1.37070747 0.65821311 1.28162837 2.32850884 0.35568269\n",
      " 5.59809385 2.56250977 0.65337851 1.66677486 4.44072504 0.67647898\n",
      " 4.2431713  2.49082802 3.58559852 3.68041696 2.14204093 0.72710782\n",
      " 1.33142775 0.07980499 2.06934767 4.81329944 0.38274976 1.44017585\n",
      " 0.517264   1.56475134 2.90166336 1.76506969 0.82489956 2.14061581\n",
      " 1.7051542  2.27531678 2.25405712 2.21596269 0.56299452 3.65007997\n",
      " 1.75460946 2.58235562 0.33957843 1.57484784 1.27658003 0.92385601\n",
      " 1.19201702 1.70547235 1.00781292 0.43665    1.41809391 1.61023568\n",
      " 0.45768766 1.00685589 0.69706675 2.2113672  1.50159621 1.73333827\n",
      " 1.46827328 0.94929531 2.08341848 2.24921757 2.89723797 1.91374622\n",
      " 2.63018927 0.24272081 1.14607885 2.65090697 0.40767526 2.17932078\n",
      " 2.32192334 0.96365435 1.97226189 1.49290258 2.32293155 3.69472609\n",
      " 2.00160869 2.2854885  2.6382113  0.84961488 1.39492863 3.03003407\n",
      " 1.58088248 5.62523416 1.71751993 2.42337301 2.75401434 0.63078396\n",
      " 1.45538902 0.50594664 1.31797236 3.06388895 1.06615409 1.69358554\n",
      " 1.29802278 4.08301611 1.56400536 2.52103876 1.2287661  2.06853854\n",
      " 2.663299   7.43561019 3.22523506 2.81834171 1.77620771 1.36814247\n",
      " 0.97051052 3.87172689 1.04427054 3.86206131 1.32823053 1.33945985\n",
      " 1.14532497 1.07035423 1.48740987 1.87630287 0.77832024 2.53679354\n",
      " 3.37542347 3.40470477 3.23500865 1.37390475 2.03945002 5.54370026\n",
      " 2.72262896 1.48480194 1.19285201 1.2835762  4.0869243  0.70674603\n",
      " 2.75962226 1.30900164 1.6942031  2.71063532 0.09912464 3.32045919\n",
      " 2.71343471 1.67504835 1.52160485 0.32445794 0.76570452 3.07421671\n",
      " 0.88185359 2.4165476  3.90439442 0.74732927 2.08553822 1.36290302\n",
      " 2.9409131  1.92611958 2.58038841 1.94228079 8.87674538 2.69671994\n",
      " 0.8234388  0.20302023 1.55230489 3.72255138 1.67136694 0.10964349\n",
      " 0.87686468 1.70367283 1.18161487 3.87211799 1.42141725 2.0225399\n",
      " 0.68431644 0.50278782 0.90398267 0.80827954 2.37258308 1.40000744\n",
      " 1.31610902 3.84145421 1.78808762 2.15854906 1.35719693 0.55632808\n",
      " 1.0081934  1.12175264 2.35319124 1.46244067 1.71002827 0.43458553\n",
      " 1.75291277 1.34147167 1.12484494 2.620355   0.50811596 4.22956916\n",
      " 2.59704308 2.07626245 5.09593159 2.43896833 2.36919204 1.58000646\n",
      " 0.79936988 0.59485707 2.69547079 2.28562445 0.86786136 0.94357945\n",
      " 2.82161017 1.1503407  2.43820496 2.43265037 1.17426911 1.40766523\n",
      " 0.66146105 3.01021515 1.33274255 1.90826893 2.17598759 2.65251567\n",
      " 1.02106954 1.86328546 2.20559574 1.47690078 1.27543046 2.48605724\n",
      " 3.20358553 0.69734168 2.34555707 2.23505155 0.69115763 2.43149856\n",
      " 2.34087573 0.73923332 1.77638456 1.49601614 3.50020534 0.34298665\n",
      " 0.9361484  1.78890034 1.87092411 8.3691953  1.48560748 2.66130281\n",
      " 4.81291202 1.01224827 2.03577853 0.99811511 2.6425345  0.83019839\n",
      " 2.20982489 0.83790665 0.41284756 4.04138428 2.22460171 0.46548483\n",
      " 3.20874863 2.2294396  1.41787494 0.70576129 8.90078212 0.67396455\n",
      " 0.80334629 0.97650148 1.54531706 0.22275952 0.7497855  1.33532056\n",
      " 2.18203084 1.23590404 2.23201329 1.35756963 1.05484645 1.25390955\n",
      " 0.58492301 1.28072481 0.81905493 2.71883247 2.9342421  0.82468347\n",
      " 2.56318736 0.4688392  1.4030321  2.06750085 1.62777469 1.92707608\n",
      " 3.22844417 1.27866242 1.20137403 1.76201855 0.69450115 5.00105201\n",
      " 4.38027536 0.64623028 4.26759504 0.88354047 4.99128584 4.25327573\n",
      " 0.75821263 0.66079587 2.27401516 1.44481767 1.85696455 0.16554111\n",
      " 1.02474888 1.18441426 0.8391342  1.3381866  2.25537498 1.33581773\n",
      " 0.17246968 1.49357169 4.17845943 0.36850787 1.53061756 1.86508261\n",
      " 2.06308152 0.39063061 0.65016076 3.53956462 0.85169271 1.66924691\n",
      " 1.43024157 1.88785725 1.64589439 1.5158603  4.36734791 0.89279767\n",
      " 0.32630566 2.17299965 0.15559669 4.4320321  1.3544467  3.05104335\n",
      " 0.91449567 1.36291051 1.97162221 1.17181025 1.33548488 3.32216471\n",
      " 3.04293342 2.69578328 2.20660977 0.76258057 1.39344183 1.25616382\n",
      " 2.48674651 3.26758111 2.23880133 1.30816811 3.13795905 0.30933935\n",
      " 3.39585718 0.18951582 2.2452448  0.66408285 1.19942756 1.51098842\n",
      " 1.53700737 3.51114842 0.18147665 9.02205091 5.1043206  2.19049297\n",
      " 3.84252028 1.13518929 3.32948234 0.53122286 2.3424423  0.0844744\n",
      " 1.35896673 1.0070684  1.63561015 0.33507399]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(array([241., 331., 212., 115.,  55.,  25.,  11.,   3.,   2.,   5.]),\n",
       " array([0.04917766, 0.97017793, 1.8911782 , 2.81217848, 3.73317875,\n",
       "        4.65417902, 5.57517929, 6.49617956, 7.41717983, 8.3381801 ,\n",
       "        9.25918038]),\n",
       " <BarContainer object of 10 artists>)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##Gamma Distribution\n",
    "Gamma_D = np.random.gamma(shape=2, scale = 1, size = 1000) # shape is mean, scale is width\n",
    "print(\"samples from Gamma Distribution are: \", Gamma_D)\n",
    "pyplot.hist(Gamma_D,color='purple')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Central limit theorem \n",
    "\n",
    "## Running the example generates and prints the sample of 50 die rolls and the mean value of the sample.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[6 4 5 1 2 4 6 1 1 2 5 6 5 2 3 5 6 3 5 4 5 3 5 6 3 5 2 2 1 6 2 2 6 2 2 1 5\n",
      " 2 1 1 6 4 3 2 1 4 6 2 2 4]\n",
      "3.44\n"
     ]
    }
   ],
   "source": [
    "# generate random dice rolls\n",
    "from numpy.random import seed\n",
    "from numpy.random import randint\n",
    "from numpy import mean\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# generate a sample of die rolls\n",
    "rolls = randint(1, 7, 50)\n",
    "print(rolls)\n",
    "print(mean(rolls))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# demonstration of the central limit theorem\n",
    "from numpy.random import seed\n",
    "from numpy.random import randint\n",
    "from numpy import mean\n",
    "from matplotlib import pyplot\n",
    "# seed the random number generator\n",
    "seed(1)\n",
    "# calculate the mean of 50 dice rolls X times\n",
    "means1 = [mean(randint(1, 7, 50)) for _ in range(10)]\n",
    "means2 = [mean(randint(1, 7, 50)) for _ in range(50)]\n",
    "means3 = [mean(randint(1, 7, 50)) for _ in range(100)]\n",
    "means4 = [mean(randint(1, 7, 50)) for _ in range(500)]\n",
    "means5 = [mean(randint(1, 7, 50)) for _ in range(2000)]\n",
    "\n",
    "\n",
    "\n",
    "# plot the distribution of sample means\n",
    "\n",
    "pyplot.hist(means5,color='purple')\n",
    "pyplot.hist(means4,color='blue')\n",
    "pyplot.hist(means3,color='green')\n",
    "pyplot.hist(means2,color='red')\n",
    "pyplot.hist(means1,color='orange')\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "#SMOTE\n",
    "# !pip install -U imbalanced-learn\n",
    "\n",
    "# check version number\n",
    "import imblearn\n",
    "# print(imblearn.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "## OVERSAMPLING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "import imblearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({0: 891, 1: 9})\n"
     ]
    },
    {
     "data": {
      "image/png": "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XX65WxAlDwYiREQ22FTbFDUTEi4AoK65DZtqm+I3KCIbMBAhIrLBoaPKQYie2xElKgYiREQ26J2bqX4jDbcjSlQMRIiIbHBRaSGKvZlQKtJ1IVg9c1FpYTyHRRR3DESIiGzgcbvwwFVlABAVjEh/f+CqMvYToaTHQISIyCYV5cVYdONI+Lzdl1983kwsunEk+4hQSmBDMyIiG1WUF2NymY+dVSllMRAhIrKZx+3CmEG97B4GkS24NENERES2YSBCREREtmEgQkRERLZhIEJERES2YSBCREREtmEgQkRERLZhIEJERES2YSBCREREtmEgQkRERLZxdGfVQCAAAGhpabF5JERERCRKOm9L5/FYHB2IHD16FADQv39/m0dCREREWh09ehRerzfmbVwBkXDFJn6/HwcOHEBubi5cLm4AFamlpQX9+/fHvn37kJeXZ/dwEgbfN334vunD900fvm/6OOV9CwQCOHr0KPr27Qu3O3YWiKNnRNxuN8444wy7h+F4eXl5/KLqwPdNH75v+vB904fvmz5OeN/UZkIkTFYlIiIi2zAQISIiItswEElgGRkZeOCBB5CRkWH3UBIK3zd9+L7pw/dNH75v+iTi++boZFUiIiJKbpwRISIiItswECEiIiLbMBAhIiIi2zAQISIiItswEEkSU6dOxYABA5CZmYni4mLcdNNNOHDggN3DcrTdu3fjlltuQWlpKbKysjBo0CA88MAD6OjosHtojvfHP/4RY8eORXZ2NvLz8+0ejmM9/vjjKC0tRWZmJkaNGoUPPvjA7iE53vvvv4+rrroKffv2hcvlwr/+9S+7h+R48+fPx4UXXojc3Fz07t0bV199Nb744gu7hyWMgUiSmDBhAl599VV88cUXeO2111BTU4PrrrvO7mE52ueffw6/348nn3wSn376KRYsWIAnnngC9913n91Dc7yOjg5873vfw4wZM+weimO98soruOuuu3D//fdj69atuPjii3H55Zdj7969dg/N0VpbWzF8+HAsXLjQ7qEkjPfeew933nknNm7ciJUrV+LkyZO47LLL0NraavfQhLB8N0ktW7YMV199Ndrb25GWlmb3cBLGn//8ZyxatAi7du2yeygJ4fnnn8ddd92FI0eO2D0Ux/nmN7+JkSNHYtGiRaGfDRs2DFdffTXmz59v48gSh8vlwhtvvIGrr77a7qEklK+//hq9e/fGe++9h0suucTu4ajijEgSampqwosvvoixY8cyCNGoubkZhYWFdg+DElxHRwc2b96Myy67rNvPL7vsMqxfv96mUVGqaG5uBoCEOZYxEEkis2fPRk5ODnr16oW9e/di6dKldg8podTU1OBvf/sbbr/9druHQgmuoaEBXV1d6NOnT7ef9+nTB/X19TaNilJBIBDA3XffjfHjx6O8vNzu4QhhIOJgc+fOhcvlivnno48+Ct3+17/+NbZu3Yp33nkHHo8HP/rRj5CKK29a3zcAOHDgACoqKvC9730Pt956q00jt5ee941ic7lc3f4eCASifkZkppkzZ2L79u1YsmSJ3UMR1sPuAZCymTNnYvr06TFvU1JSEvr/oqIiFBUVYejQoRg2bBj69++PjRs3YsyYMRaP1Fm0vm8HDhzAhAkTMGbMGDz11FMWj865tL5vpKyoqAgejydq9uPQoUNRsyREZvnFL36BZcuW4f3338cZZ5xh93CEMRBxMCmw0EOaCWlvbzdzSAlBy/u2f/9+TJgwAaNGjcLixYvhdqfuJKGRzxt1l56ejlGjRmHlypW45pprQj9fuXIlpk2bZuPIKBkFAgH84he/wBtvvIH/+7//Q2lpqd1D0oSBSBLYtGkTNm3ahPHjx6OgoAC7du3C7373OwwaNCjlZkO0OHDgAC699FIMGDAAjzzyCL7++uvQv/l8PhtH5nx79+5FU1MT9u7di66uLmzbtg0AMHjwYPTs2dPewTnE3XffjZtuugkXXHBBaLZt7969zEFScezYMezcuTP099raWmzbtg2FhYUYMGCAjSNzrjvvvBMvvfQSli5ditzc3NBMnNfrRVZWls2jExCghLd9+/bAhAkTAoWFhYGMjIxASUlJ4Pbbbw989dVXdg/N0RYvXhwAIPuHYvvxj38s+76tWbPG7qE5ymOPPRYYOHBgID09PTBy5MjAe++9Z/eQHG/NmjWyn60f//jHdg/NsZSOY4sXL7Z7aELYR4SIiIhsk7oL4kRERGQ7BiJERERkGwYiREREZBsGIkRERGQbBiJERERkGwYiREREZBsGIkRERGQbBiJERERkGwYiREREZBsGIkRERGQbBiJERERkGwYiREREZJv/H34YjZU6hpRFAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({0: 891, 1: 9})\n"
     ]
    }
   ],
   "source": [
    "# Generate and plot a synthetic imbalanced classification dataset\n",
    "from collections import Counter\n",
    "from sklearn.datasets import make_classification\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.pipeline import Pipeline\n",
    "from matplotlib import pyplot\n",
    "from numpy import where\n",
    "# define dataset\n",
    "X, y = make_classification(n_samples=900, n_features=2, n_redundant=0,\n",
    " n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=10)\n",
    "# summarize class distribution\n",
    "counter = Counter(y)\n",
    "print(counter)\n",
    "# scatter plot of examples by class label\n",
    "for label, _ in counter.items():\n",
    " row_ix = where(y == label)[0]\n",
    " pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))\n",
    "pyplot.legend()\n",
    "pyplot.show()\n",
    "# summarize class distribution\n",
    "counter = Counter(y)\n",
    "print(counter)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "#SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({0: 891, 1: 891})\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# transform the dataset\n",
    "oversample = SMOTE()\n",
    "X, y = oversample.fit_resample(X, y)\n",
    "# summarize the new class distribution\n",
    "counter = Counter(y)\n",
    "print(counter)\n",
    "# scatter plot of examples by class label\n",
    "for label, _ in counter.items():\n",
    " row_ix = where(y == label)[0]\n",
    " pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))\n",
    "pyplot.legend()\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Adasyn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Counter({0: 891, 1: 891})\n",
      "Counter({0: 891, 1: 891})\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from imblearn.over_sampling import ADASYN\n",
    "# summarize class distribution\n",
    "counter = Counter(y)\n",
    "print(counter)\n",
    "# transform the dataset\n",
    "oversample = ADASYN()\n",
    "X, y = oversample.fit_resample(X, y)\n",
    "# summarize the new class distribution\n",
    "counter = Counter(y)\n",
    "print(counter)\n",
    "# scatter plot of examples by class label\n",
    "for label, _ in counter.items():\n",
    " row_ix = where(y == label)[0]\n",
    " pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))\n",
    "pyplot.legend()\n",
    "pyplot.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
