#!/usr/bin/python3

import sys
import numpy as np

# path to price data
path = "./Data/"

# list of stock tickers to trade
# with corresponding ticker.txt file 
# in the directory specified by path 
tradelist = ["aapl", "nflx"]

# add checker for unanimity in file lengths

# choices for lengths
# fix this to be determined by unanimous file lengths
# daycount = 1257 ## 6 yrs
# daycount = 1006 ## 5 yrs
# daycount = 752  ## 3 yrs
daycount = 251  ## 1 yr

# matrix of daily stock prices
prices = np.zeros((len(tradelist), daycount))

def data_fetch():
    count = 0
    for filename in tradelist:
        filepath = path + filename + ".txt"
        f = open(filepath)
        lines = (f.readlines())
        lines = lines[::-1]
        prices[count, :] = [float(x) for x in lines]
        count += 1
        f.close()

# matrix of daily stock returns
returns = np.zeros((len(tradelist), daycount + 1))

def data_prep():
    global returns
    # need to push the future foreward so time lags back
    backward = np.concatenate([np.ones((len(tradelist), 1)), prices], 1)
    # make the matrices equally-sized
    forward = np.concatenate([prices, np.ones((len(tradelist), 1))], 1)

    # as per Thomas Cover, we use (today / yesterday)-style returns
    invback = [ (1.0 / x) for x in backward]
    returns = np.multiply(forward, invback)

    # get rid of the garbage rows: the first and last
    returns = np.delete(returns, (0, daycount), 1)
    
# takes an integer n, with 1 <= n <= daycount
# i.e., takes any day for which the stock price has been monitored
# and computes the cumulative return of the constant (reallocated) portfolio 
# whose weights are specified in the (len(tradelist) x 1)-dimensional array b
def S(n, b):
    global returns
    stb = 1.0
    # for every return since returns[:, 0] (info avail from prices[:,1])
    # until the n-th trading day; setup on day 0, hold til day1, make r(0)
    # so S(1) = b * r(0)
    # which calc'd by looking at the first n returns, i.e., r(0),..., r(n-1)
    for t in range(1, n + 1):
        # this weirdness is for consistency with emp_S_hat(n)'s for-loop, (fear not!)
        xt = returns[:, t - 1]
        bport_t = 0
        for e in range(0, len(tradelist)):
            bport_t += b[e] * xt[e]
        stb *= bport_t
    return stb

# the universal portfolio weights for day k. 
# The current implementation will only work for two stocks 
def emp_b(k):
    if k == 0:
        print("emp_b error: undefined on day 0")
        exit(0)
    if k == 1:
        b1 = np.ones((len(tradelist), 1))
        pie = (1.0 / float(len(tradelist)))
        b1 = pie * b1
        return b1
    quantization = 20
    numer = 0
    denom = 0
    # on page 20 of cover's paper, integral shows b(k+1) as an expr of Sk
    # in expression (8.2)
    # but the expression in (8.4) is b(k) in terms of Sk, which uses info
    # about the returns of the kth trading day, i.e., future info
    # is built into the weighting; adjust to say S(k-1) on RHS of (8.4).
    for i in range(0, quantization + 1):
        portion = (float(i) / float(quantization))
        b_form_of_i = np.array([portion, (1.0 - portion)])
        ski = S(k-1, b_form_of_i)
        numer += (portion * ski)
        denom += ski
    val = (float(numer) / float(denom))
    # hyperadaptive version: for comparison purposes only
    # if (val < 0.50):
    #    val = float(val / 2.0)
    # else:
    #    val = float((val + 1.0) / 2.0)
    return np.array([val, (1.0 - val)])

# the wealth of the universal portfolio on day n,
# as determined by the empirically-calculated b(k)'s.
def emp_S_hat(n):
    global returns
    total_rtn = 1
    day_rtn = 0

    # these are the total return contributions for each stock
    # as weighted by the corresponding weights in b(k) for each k <= n.
    tot_0 = 1
    tot_1 = 1

    wealth = np.ones((n, 1))

    # include days 1 thru n, (n+1) is excluded.
    for k in range(1, n + 1):
        day_rtn = 0
        bk = emp_b(k)
        for e in range(0, len(tradelist)):
            # b1 is the sensitivity to r(0), so need to backshift returns when indexing
            day_rtn += float(bk[e] * returns[e, k - 1])
        # for tracking purposes, check if Univ beats both others
        tot_0 *= returns[0, k - 1]
        tot_1 *= returns[1, k - 1]
        # univ bookkeeping
        total_rtn *= day_rtn

        wealth[k - 1] = total_rtn
        
        # pretty-printing, sort of...
        spc = ""
        if k < 10:
            spc = "0"

        dayinfo = "Day " + spc + str(k) + ":\t "
        portinfo = "Univ  " + str(total_rtn) + "\t "
        stock1info = tradelist[0] + "  " + str(bk[0]) + "  " + str(tot_0) + "\t " 
        stock2info = tradelist[1] + "  " + str(bk[1]) + "  " + str(tot_1)
        print(dayinfo + portinfo + stock1info + stock2info)


def main():
    data_fetch()
    data_prep()
    
    # Six years:
    # emp_S_hat(1256)
    
    # 5 years: 
    # emp_S_hat(1005)

    # 3 years:
    emp_S_hat(751)
    
    # 1 year:
    # emp_S_hat(250)

main()
