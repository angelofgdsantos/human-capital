import numpy as np
import kw_functions as kw 
'''

Parameters

'''
age_start = 16
age_max   = 30
'''

For last age

'''
cells = np.empty((age_max,age_max,age_max,age_max,age_max))
cells[:] = np.nan  

for a in range(age_max, -1, 1):
    # White color possibilities
    for w in range(age_max):
        if w == age_max-1:
            x = np.array([w,0,0,0,0])
            cells[w][0][0][0][0] = kw.R_m(a,0,x)[0]
        else:
            # Blue color possibilities
            for b in range(age_max):
                if b == age_max-1:
                    x = np.array([0,b,0,0,0])
                    cells[0][b][0][0][0] = kw.R_m(a,0,x)[1]
                elif w + b == age_max - 1:
                    x = np.array([w,b,0,0,0])
                    cells[w][b][0][0][0] = kw.R_m(a,0,x)[1]                  
                else:
                    # Military possibilities
                    for m in range(age_max):
                        if m == age_max-1:
                            x = np.array([0,0,m,0,0])
                            cells[0][0][m][0][0] = kw.R_m(a,0,x)[2] 
                        elif w + b + m == age_max - 1:
                            x = np.array([w,b,m,0,0])
                            cells[w][b][m][0][0] = kw.R_m(a,0,x)[2]                          
                        else:
                            # Schooling possibilities
                            for s in range(age_max):
                                if s == age_max-1:
                                    x = np.array([0,0,0,s,0])
                                    cells[0][0][0][s][0] = kw.R_m(a,s,x)[3] 
                                elif w + b + m + s == age_max - 1:
                                    x = np.array([w,b,m,s,0])
                                    cells[w][b][m][s][0] = kw.R_m(a,s,x)[3]  
                                else:
                                    # Housing possibilities
                                    for h in range(age_max):
                                        if h == age_max-1:
                                            x = np.array([0,0,0,0,h])
                                            cells[0][0][0][0][h] = kw.R_m(a,0,x)[4] 
                                        elif w + b + m + s + h == age_max - 1:
                                            x = np.array([w,b,m,s,h])
                                            cells[w][b][m][s][h] = kw.R_m(a,s,x)[4] 
                                        else:
                                            x = np.array([w,b,m,s,h])
                                            cells[w][b][m][s][h] = kw.R_m(a,s,x)[4] 

                                            

                                            
