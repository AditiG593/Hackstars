# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:45:19 2025

@author: Admin
"""

import cleaning
import merge
import lag
import scale
import filled_data

cleaning.clean_data()
merge.merge()
lag.lag_fun()
filled_data.fun1_fill()