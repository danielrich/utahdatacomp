%This function was written by Ben Taylor @bentaylordata to demonstrate two
%approaches to this problem (LSR, and EWLSR). The first regression ignores
%the time effects, the second weights them with a 0.8 lambda decay, think
%EWMA. Tweet me @bentaylordata with hashtag #utahdatacompetition with any
%questions. 

context_map=csvread('train_X.csv');             % Load train input (context matrix)
sim_yield_die_loss=csvread('train_Y.csv');      % Load train output
wafer_count=size(context_map,1);                % Get number of wafers

%Build regular LSR model +++++++++++++++++++++++++++++++++++++++++++++++++
beta_LSR=pinv(context_map'*context_map)*context_map'*sim_yield_die_loss;   %regular LSR model


% Now load validation set and score
context_map_val=csvread('val_X.csv');         %Load train input (context matrix)
sim_yield_val_die_loss=csvread('val_Y.csv');           %Load train output

%Evaluate model
yp_LSR=context_map_val*beta_LSR;                   %Predict regression

%Make sure to cap crazy results outside possible die >600 <0
yp_LSR(yp_LSR<0)=0;
yp_LSR(yp_LSR>600)=600;

%Special cost function, under prediction is 10x penalty
err_LSR=yp_LSR-sim_yield_val_die_loss;              %Calculate errors

%Custom residual sum
err_underpredict_LSR=abs(err_LSR(err_LSR<0))*10;        %10x penalty
err_overpredict_LSR=abs(err_LSR(err_LSR>=0));

overall_score_LSR=(sum(err_underpredict_LSR)+sum(err_overpredict_LSR))/wafer_count   
