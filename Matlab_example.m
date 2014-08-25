%This function was written by Ben Taylor @bentaylordata to demonstrate two
%approaches to this problem (LSR, and EWLSR). The first regression ignores
%the time effects, the second weights them with a 0.8 lambda decay, think
%EWMA. Tweet me @bentaylordata with hashtag #utahdatacompetition with any
%questions. 

context_map=csvread('train_X.csv');         %Load train input (context matrix)
sim_yield=csvread('train_Y.csv');           %Load train output

sim_yield_die_loss=600-sim_yield;           %Train towards die loss

wafer_count=size(context_map,2);            %Get number of wafers

%Build regular LSR model +++++++++++++++++++++++++++++++++++++++++++++++++
beta_LSR=pinv(context_map*context_map')*context_map*sim_yield_die_loss';

%Build WLSR model ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dim=size(context_map,2);                    %Get number of observations
lambda=0.8;                                 %Decay rate on learning
w_diag=fliplr(lambda.^(1:dim));
W=zeros(dim,dim);
for g=1:dim
    W(g,g)=w_diag(g);
end
beta_WLSR=pinv(context_map*W*context_map')*context_map*W*sim_yield_die_loss';

% Now load validation set and score
context_map_val=csvread('val_X.csv');         %Load train input (context matrix)
sim_yield_val=csvread('val_Y.csv');           %Load train output

sim_yield_val_die_loss=600-sim_yield_val;           %Train towards die loss

%Evaluate models
yp_LSR=beta_LSR'*context_map_val;                   %Predict regression
yp_WLSR=beta_WLSR'*context_map_val;                 

%Special cost function, under prediction is 10x penalty
err_LSR=yp_LSR-sim_yield_val_die_loss;              %Calculate errors
err_WLSR=yp_WLSR-sim_yield_val_die_loss;

%Custom residual sum
err_underpredict_LSR=abs(err_LSR(err_LSR<0))*10;        %10x penalty
err_overpredict_LSR=abs(err_LSR(err_LSR>=0));
err_underpredict_WLSR=abs(err_WLSR(err_WLSR<0))*10;        %10x penalty
err_overpredict_WLSR=abs(err_WLSR(err_WLSR>=0));

overall_score_LSR=(sum(err_underpredict_LSR)+sum(err_overpredict_LSR))/wafer_count;   
overall_score_WLSR=(sum(err_underpredict_WLSR)+sum(err_overpredict_WLSR))/wafer_count;
