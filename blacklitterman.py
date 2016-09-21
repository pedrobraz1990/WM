# Black-Litterman example code for python (hl.py)
# Copyright (c) Jay Walters, blacklitterman.org, 2012.
#
# Redistribution and use in source and binary forms, 
# with or without modification, are permitted provided 
# that the following conditions are met:
#
# Redistributions of source code must retain the above 
# copyright notice, this list of conditions and the following 
# disclaimer.
# 
# Redistributions in binary form must reproduce the above 
# copyright notice, this list of conditions and the following 
# disclaimer in the documentation and/or other materials 
# provided with the distribution.
#  
# Neither the name of blacklitterman.org nor the names of its
# contributors may be used to endorse or promote products 
# derived from this software without specific prior written
# permission.
#  
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR 
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.
#
# This program uses the examples from the paper "The Intuition 
# Behind Black-Litterman Model  Portfolios", by He and Litterman,
# 1999.  You can find a copy of this  paper at the following url.
#     http://papers.ssrn.com/sol3/papers.cfm?abstract_id=334304
#
# For more details on the Black-Litterman model you can also view
# "The BlackLitterman Model: A Detailed Exploration", by this author
# at the following url.
#     http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1314585
#

import numpy as np
from scipy import linalg

# blacklitterman
#   This function performs the Black-Litterman blending of the prior
#   and the views into a new posterior estimate of the returns as
#   described in the paper by He and Litterman.
# Inputs
#   delta  - Risk tolerance from the equilibrium portfolio
#   weq    - Weights of the assets in the equilibrium portfolio
#   sigma  - Prior covariance matrix
#   tau    - Coefficiet of uncertainty in the prior estimate of the mean (pi)
#   P      - Pick matrix for the view(s)
#   Q      - Vector of view returns
#   Omega  - Matrix of variance of the views (diagonal)
# Outputs
#   Er     - Posterior estimate of the mean returns
#   w      - Unconstrained weights computed given the Posterior estimates
#            of the mean and covariance of returns.
#   lambda - A measure of the impact of each view on the posterior estimates.
#
def blacklitterman(delta, weq, sigma, tau, P, Q, Omega):
  # Reverse optimize and back out the equilibrium returns
  # This is formula (12) page 6.
  pi = weq.dot(sigma * delta)
  print(pi)
  # We use tau * sigma many places so just compute it once
  ts = tau * sigma
  # Compute posterior estimate of the mean
  # This is a simplified version of formula (8) on page 4.
  middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
  print(middle)
  print(Q-np.expand_dims(np.dot(P,pi.T),axis=1))
  er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
  # Compute posterior estimate of the uncertainty in the mean
  # This is a simplified and combined version of formulas (9) and (15)
  posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
  print(posteriorSigma)
  # Compute posterior weights based on uncertainty in mean
  w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
  # Compute lambda value
  # We solve for lambda from formula (17) page 7, rather than formula (18)
  # just because it is less to type, and we've already computed w*.
  lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)
  return [er, w, lmbda]
