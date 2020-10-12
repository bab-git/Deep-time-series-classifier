#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:03:02 2020

@author: bhossein
"""
import numpy as np

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="small-problem", sense=LpMaximize)

# Initialize the decision variables: x is integer, y is continuous
x = LpVariable(name="x", lowBound=0, cat="Integer")
y = LpVariable(name="y", lowBound=0)

# Add the constraints to the model
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Add the objective function to the model
model += lpSum([x, 2 * y])

# Solve the problem
status = model.solve()

% ======= toy data
n=11;  
m = 8;
N = 100;
x = randi(2,N,n)-1;
x(x==0)=-1;
x = [x zeros(N,m)];
x(1:50,n+1:end) = ones(50,m);
x(51:N,n+1:end) = -ones(50,m);
n = n+m;

% ========  Integer problem formulation
ip = find(sign(sum(x,2))>0); 
A =-x(ip,:);
A = [A ; x(find(sign(sum(x,2))<0),:)];
b = -ones(N,1);
lb = zeros(n,1);
ub = ones(n,1);
I0 = ones(n,1);
intcon = [1:n];
f = ones(n,1);
% func = @(I)sum(I);
find(A*I0-b>0) % testing the formulated constraints: all should be negative
f'*I0  % objective

% ======== solver
I = intlinprog(f,intcon,A,b,[],[],lb,ub,I0);
I'