'''

Author: Angelo Santos

These functions are used to plot the female shares by genre and subgenres
using the LOC dataset.

REMEMBER TO CHANGE THE DIRECTORY FOR BOOKS !!!

'''

import os
from turtle import title

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

'''

Plotting settings

'''
plt.rcParams['figure.dpi'] = 500
plt.rcParams['savefig.dpi'] = 500
sns.set_style('ticks')
sns.despine(left=False, bottom=True)
sns.set_context("paper")
'''

Functions

'''
def female_share(start = 1980, end = 2011,g = 'Actual', user = 'angelosantos'):
    """
    
    This code creates a DataFrame with female shares for three groups:
        - All books: Default
        - Fiction: g = 'Fiction'
        - Non Fiction: g = 'non Fiction'

    Args:
        start (int, optional): Starting year. Defaults to 1980.
        end (int, optional): Last year. Defaults to 2011.
        g (str, optional): Group of books: 'Actual', 'Fiction', 'non Fiction'.
                            Defaults to 'Actual'.
        user (str, optional): User identifier to get the right directory. Jordan: 'jcholbro'

    Returns:
        DataFrame: DataFrame withe the female shares for group g per year,
                    between start and end.
                    
    """
    if user == 'jcholbro':
        os.chdir(r'C:\Users\jcholbro\University Of Houston\Books Project - Generaldata\data_created\full_datasets\loc\merged')
    os.chdir('/Users/'+user+'/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/data/data_created/full_datasets/loc/merged')
    books = pd.read_pickle('cre_authors_books_v1.pkl')
    fr = []
    for y in range(start,end):
        fs = {}
        baseline = books.loc[books['pub_year'] == y]                           
        if g == 'Fiction':                                                     
            baseline = baseline.loc[baseline['fic_score'] == 1]
        if g == 'non Fiction':                                                        
            baseline = baseline.loc[baseline['fic_score'] == 0]
        else:
            pass
        fem = baseline[baseline['gender_combined'] == 'female']          
        fem_share = fem.shape[0]/baseline.shape[0]
        fs['year'] = y
        fs['fs'] = fem_share
        fs['genre'] = g
        fr.append(fs)
    frame = pd.DataFrame(fr)
    return frame

def genre_share_fic(start = 1980, end = 2011, fiction = 1, user = 'angelosantos'):
    """
    
    This code creates a DataFrame with genres shares for Fiction and non Fiction

    Args:
        start (int, optional): Starting year. Defaults to 1980.
        end (int, optional): Last year. Defaults to 2011.
        fiction (int, optional): Fiction or Non Fiction genres.
        user (str, optional): User identifier to get the right directory. Jordan: 'jcholbro'

    Returns:
        DataFrame: DataFrame withe the genre shares for Fiction or non Fiction per year,
                    between start and end.
                   
    """
    if user == 'jcholbro':
        os.chdir(r'C:\Users\jcholbro\University Of Houston\Books Project - Generaldata\data_created\full_datasets\loc\merged')
    os.chdir('/Users/'+user+'/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/data/data_created/full_datasets/loc/merged')
    books = pd.read_pickle('cre_authors_books_v1.pkl')
    fr = []
    for y in range(start,end):
        fs = {}
        baseline = books.loc[books['pub_year'] == y]                           
        if fiction == 1:                                                      
            fic = baseline.loc[baseline['fic_score'] == 1]
            g = 'Fiction'
        else:
            fic = baseline.loc[baseline['fic_score'] == 0]
            g = 'non Fiction'
        genre_share = fic.shape[0]/baseline.shape[0]
        fs['year'] = y
        fs['fs'] = genre_share
        fs['genre'] = g
        fr.append(fs)
    frame = pd.DataFrame(fr)
    return frame
   

def subg_female_share(start = 1980, end = 2011, fiction = 1, g = 'Romance', user = 'angelosantos'):
    """
    
    This code creates a DataFrame with female shares for subgenres in fiction and non Fiction

    Args:
        start (int, optional): Starting year. Defaults to 1980.
        end (int, optional): Last year. Defaults to 2011.
        fiction (int, optional): Fiction or Non fiction genres.
        g (str, optional): Subgenre in Fiction or non Fiction:
                            Fiction: 
                                - 'Action/Adventure'
                                - 'Childrens Stories'
                                - 'Fantasty/Sci-Fi'
                                - 'Horror/Paranormal' 
                                - 'Mystery/Crime' 
                                - 'Romance'
                                - 'Suspence'
                                - 'Spy/Politics'
                                - 'Literary_1'
                                
                            Non Fiction:
                                - 'LCC_gen_AGRICULTURE'
                                - 'LCC_gen_AUXILIARY SCIENCES OF HISTORY'
                                - 'LCC_gen_BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES (GENERAL)'
                                - 'LCC_gen_EDUCATION'
                                - 'LCC_gen_FINE ARTS'
                                - 'LCC_gen_GENERAL WORKS'
                                - 'LCC_gen_GEOGRAPHY. ANTHROPOLOGY. RECREATION'
                                - 'LCC_gen_HISTORY OF THE AMERICAS'
                                - 'LCC_gen_LAW'
                                - 'LCC_gen_MEDICINE'
                                - 'LCC_gen_MILITARY SCIENCE'
                                - 'LCC_gen_MUSIC AND BOOKS ON MUSIC'
                                - 'LCC_gen_NAVAL SCIENCE'
                                - 'LCC_gen_PHILOSOPHY. PSYCHOLOGY. RELIGION', 'LCC_gen_POLITICAL SCIENCE'
                                - 'LCC_gen_SCIENCE'
                                - 'LCC_gen_SOCIAL SCIENCES'
                                - 'LCC_gen_TECHNOLOGY'
                                - 'LCC_gen_WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, NEW ZEALAND, ETC.'

        user (str, optional): User identifier to get the right directory. Jordan: 'jcholbro'

    Returns:
        DataFrame: DataFrame with the female shares for Fiction or non Fiction for subgenre g per year,
                    between start and end.
                   
    """
    if user == 'jcholbro':
        os.chdir(r'C:\Users\jcholbro\University Of Houston\Books Project - Generaldata\data_created\full_datasets\loc\merged')
    os.chdir('/Users/'+user+'/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/data/data_created/full_datasets/loc/merged')
    books = pd.read_pickle('cre_authors_books_v1.pkl')
    fr = []
    for y in range(start,end):
        fs = {}
        baseline = books.loc[books['pub_year'] == y]                           
        if fiction == 1:                                                      
            fic_base = baseline.loc[baseline['fic_score'] == 1]
            fic_base = fic_base.loc[fic_base['fic_counts'] < 2]
            fic = fic_base.loc[fic_base[g] == 1]
            fem_fic = fic.loc[fic['gender_combined'] == 'female']
        else:
            fic_base = baseline.loc[baseline['fic_score'] == 0]
            fic = fic_base.loc[fic_base[g] == 1]
            fem_fic = fic.loc[fic['gender_combined'] == 'female']
        fem_share = fem_fic.shape[0]/fic.shape[0]
        fs['year'] = y
        fs['fs'] = fem_share
        fs['genre'] = g
        fr.append(fs)
    frame = pd.DataFrame(fr)
    return frame

def subgenre_share(start = 1980, end = 2011, fiction = 1, g = 'Romance', user = 'angelosantos'):
    """
    
    This code creates a DataFrame with subgenre shares for subgenres in fiction and non Fiction

    Args:
        start (int, optional): Starting year. Defaults to 1980.
        end (int, optional): Last year. Defaults to 2011.
        fiction (int, optional): Fiction or Non fiction genres.
        g (str, optional): Subgenre in Fiction or non Fiction:
                            Fiction: 
                                - 'Action/Adventure'
                                - 'Childrens Stories'
                                - 'Fantasty/Sci-Fi'
                                - 'Horror/Paranormal' 
                                - 'Mystery/Crime' 
                                - 'Romance'
                                - 'Suspence'
                                - 'Spy/Politics'
                                - 'Literary_1'
                                
                            Non Fiction:
                                - 'LCC_gen_AGRICULTURE'
                                - 'LCC_gen_AUXILIARY SCIENCES OF HISTORY'
                                - 'LCC_gen_BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES (GENERAL)'
                                - 'LCC_gen_EDUCATION'
                                - 'LCC_gen_FINE ARTS'
                                - 'LCC_gen_GENERAL WORKS'
                                - 'LCC_gen_GEOGRAPHY. ANTHROPOLOGY. RECREATION'
                                - 'LCC_gen_HISTORY OF THE AMERICAS'
                                - 'LCC_gen_LAW', 'LCC_gen_MEDICINE'
                                - 'LCC_gen_MILITARY SCIENCE'
                                - 'LCC_gen_MUSIC AND BOOKS ON MUSIC'
                                - 'LCC_gen_NAVAL SCIENCE'
                                - 'LCC_gen_PHILOSOPHY. PSYCHOLOGY. RELIGION', 'LCC_gen_POLITICAL SCIENCE'
                                - 'LCC_gen_SCIENCE'
                                - 'LCC_gen_SOCIAL SCIENCES'
                                - 'LCC_gen_TECHNOLOGY'
                                - 'LCC_gen_WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, NEW ZEALAND, ETC.'

        user (str, optional): User identifier to get the right directory. Jordan: 'jcholbro'

    Returns:
        DataFrame: DataFrame with the genre shares for Fiction or non Fiction for subgenre g per year,
                    between start and end.
                   
    """
    if user == 'jcholbro':
        os.chdir(r'C:\Users\jcholbro\University Of Houston\Books Project - Generaldata\data_created\full_datasets\loc\merged')
    os.chdir('/Users/'+user+'/Library/CloudStorage/OneDrive-SharedLibraries-UniversityOfHouston/Books Project - General/data/data_created/full_datasets/loc/merged')
    books = pd.read_pickle('cre_authors_books_v1.pkl')
    fr = []
    for y in range(start,end):
        fs = {}
        baseline = books.loc[books['pub_year'] == y]                           
        if fiction == 1:                                                      
            fic_base = baseline.loc[baseline['fic_score'] == 1]
            fic_base = fic_base.loc[fic_base['fic_counts'] < 2]
            fic = fic_base.loc[fic_base[g] == 1]
        else:
            fic_base = baseline.loc[baseline['fic_score'] == 0]
            fic = fic_base.loc[fic_base[g] == 1]
        genre_share = fic.shape[0]/fic_base.shape[0]
        fs['year'] = y
        fs['fs'] = genre_share
        fs['genre'] = g
        fr.append(fs)
    frame = pd.DataFrame(fr)
    return frame
