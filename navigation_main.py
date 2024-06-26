from navigation_utils import *


# -------------------- MAIN ------------------------------------------------------------------------
'''
Run this script to run a dead reckoning on the data in dir_name, both with the bias and removing it (assuming you have
calibration data for this user, otherwise just use the dir_name supplied here)
'''


#dir_name = 'walking/inhand-28-steps-Ido'
#dir_name = 'walking/inhand-20-steps-zyf'
dir_name = 'walking/inhand-20-steps-Cxh'

def main():
    dead_reckon(dir_name, remove_bias=False, title='Results plot with bias')    #包含偏差
    dead_reckon(dir_name, remove_bias=True, title='Results plot without bias', sma=0)   #不包含偏差


if __name__ == "__main__":
    main()
