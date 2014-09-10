%This function was written by Ben Taylor @bentaylordata to demonstrate two
%approaches to this problem (LSR, and EWLSR). The first regression ignores
%the time effects, the second weights them with a 0.8 lambda decay, think
%EWMA. Tweet me @bentaylordata with hashtag #utahdatacompetition with any
%questions. 

context_map=csvread('train_X.csv');             % Load train input (context matrix)
sim_yield_die_loss=csvread('train_Y.csv');      % Load train output
wafer_count=size(context_map,1);                % Get number of wafers

%Build regular WLSR model +++++++++++++++++++++++++++++++++++++++++++++++++
lambda=0.8;                                 %Decay rate on learning
w_diag=fliplr(lambda.^(1:wafer_count));
W=zeros(wafer_count);
for g=1:dim
    W(g,g)=w_diag(g);
end
beta_WLSR=pinv(context_map'*W*context_map)*context_map'*W*sim_yield_die_loss;

context_map_val=csvread('val_X.csv');         %Load train input (context matrix)
sim_yield_val_die_loss=csvread('val_Y.csv');           %Load train output

%Evaluate models
yp_WLSR=context_map_val*beta_WLSR;                 

%Make sure to cap crazy results outside possible die >600 <0
yp_WLSR(yp_WLSR<0)=0;
yp_WLSR(yp_WLSR>600)=600;

%Special cost function, under prediction is 10x penalty
err_WLSR=yp_WLSR-sim_yield_val_die_loss;

%Custom residual sum
err_underpredict_WLSR=abs(err_WLSR(err_WLSR<0))*10;        %10x penalty
err_overpredict_WLSR=abs(err_WLSR(err_WLSR>=0));
  
overall_score_WLSR=(sum(err_underpredict_WLSR)+sum(err_overpredict_WLSR))/wafer_count
