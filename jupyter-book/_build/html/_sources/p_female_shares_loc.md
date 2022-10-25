---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Plotting female shares 

In this routine we will plot the female share for the [Library of Congress](https://www.loc.gov) dataset. We will plot two graphs:

1. Female share for fiction, non-fiction and all books
2. Two scale graphs, with female share (left y-axis) and genre share (right x-axis)

## Packages needed

```{code-block}
import os
import numpy as np
import pandas as pd
import seaborn as sns
from myst_nb import glue
from matplotlib import pyplot as plt
import p_func_female_shares_loc as pf 
```

```{warning}
**p_func_female_shares_loc** is a file with the functions used here.
```

## Plotting  female shares 
### 1. Plotting for actual, fiction and non-fiction

In this section, we will plot the female shares for each year, dividing in three categories:

1. **Actual:** Female shares using all books
2. **Fiction:** Female shares in the fiction books samples
3. **Non-Fiction:** Female shares in the non-fiction books samples

We will plot these shares for the 1930-2010.

```{code-block}
start = 1930
end = 2010
```

Now, we will use the function female_share using `start` and `end` as arguments to get the range we want. In addition to it, we will loop this on the three samples: `'Actual','Fiction','non Fiction'`

```{code-block}
fm = []
for i in ['Actual','Fiction','non Fiction']:
    fs = pf.female_share(start=start, end = end, g=i)
    fm.append(fs)
fm = pd.concat(fm, axis = 0).reset_index().drop('index', axis = 1)
fm['fs'] = fm['fs']*100
fm = fm.rename(columns = {
    'year': 'Year',
    'fs': 'Female Share',
    'genre' : 'Category'
    })
```

Now, we can plot the following figure. 

```
sns.lineplot(data = fm, x = fm.columns[0], y = fm.columns[1], hue = fm.columns[2], palette = 'icefire')\
    .set(title = 'Female share between '+str(start)+ ' and ' + str(end))
sns.despine(left=False, bottom=False)
os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/outputs/plots/shares/loc/female_share')
plt.savefig('p_female_shares.png')
plt.close()
})
```


### 2. Plotting two scales - One plot

In this section, we will plot the female shares and the genre share for each year. We will plot these values in two scales graph, where the left y-axis is female share and the right x-axis is the genre share. 

Firs, we create a list with all the genres that we have use to categorize the fiction books.
```{code-block}
fics = [
        'Action/Adventure', 
        'Childrens Stories', 
        'Fantasty/Sci-Fi', 
        'Horror/Paranormal', 
        'Mystery/Crime',
        'Romance', 'Suspence', 
        'Spy/Politics', 
        'Literary_1'
        ]
```

To plot the two scales, we did a unique figure with all the genres and other figures with individual genres. The first code creates the figure with all the grapphs together.

```{code-block}
i1 = 0 
i2 = 0
fig, axs = plt.subplots(3, 3, constrained_layout=True)

for f in fics:
    df1 = pf.subg_female_share(g = f)
    df2 = pf.subgenre_share(g = f)
    #define colors to use
    col1 = 'g'
    col2 = 'b'

    #define subplots
    #add first line to plot
    axs[i1][i2].plot(df1.year, df1.fs, color=col1)

    #add x-axis label
    if (i2 == 1) & (i1 == 2):
        axs[i1][i2].set_xlabel('Year', fontsize=10)

    #add y-axis label
    if (i2 == 0) & (i1 == 1):
        axs[i1][i2].set_ylabel('Female Share', color=col1, fontsize=10)

    #define second y-axis that shares x-axis with current plot
    ax2 = axs[i1][i2].twinx()

    #add second line to plot
    ax2.plot(df2.year, df2.fs, color=col2)

    #add second y-axis label
    if (i2 == 2) & (i1 == 1):
        ax2.set_ylabel('Genre Share', color=col2, fontsize=10)
    axs[i1][i2].set_title(f, fontsize=10)
    i2 = i2 + 1
    if i2 == 3:
        i2 = 0
    else: 
        pass
    if i2 == 0:
        i1 += 1
    else:
        pass
        
plt.suptitle('Two scales graphs')
os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/outputs/plots/shares/loc/two_scale')
plt.savefig('p_2sca.png')
plt.close()
```

### Plotting two scales - Separated plots

The following, plots a figure for each fiction genre. 

```{code-block}
fics = [
        'Action/Adventure', 
        'Childrens Stories', 
        'Fantasty/Sci-Fi', 
        'Horror/Paranormal', 
        'Mystery/Crime',
        'Romance', 'Suspence', 
        'Spy/Politics', 
        'Literary_1'
        ]

for f in fics:
    df1 = pf.subg_female_share(g = f)
    df2 = pf.subgenre_share(g = f)

    fig, axs = plt.subplots()
    #define colors to use
    col1 = 'g'
    col2 = 'b'

    #define subplots
    #add first line to plot
    axs.plot(df1.year, df1.fs, color=col1)

    #add x-axis label
    axs.set_xlabel('Year', fontsize=10)

    #add y-axis label
    axs.set_ylabel('Female Share', color=col1, fontsize=10)

    #define second y-axis that shares x-axis with current plot
    ax2 = axs.twinx()

    #add second line to plot
    ax2.plot(df2.year, df2.fs, color=col2)

    #add second y-axis label
    ax2.set_ylabel('Genre Share', color=col2, fontsize=10)
    axs.set_title(f, fontsize=10)
    os.chdir('/Users/angelosantos/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/outputs/plots/shares/loc/two_scale')
    plt.savefig('p_2sca_'+f.replace('/','_').replace(' ','_')+'.png')
    plt.close()
```








