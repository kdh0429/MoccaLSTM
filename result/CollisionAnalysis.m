clear all;
clc
format long
%%
cd ../data/
residual_idx = load('ResiIdx.csv');
Max20thResidual = load('ResiMax.csv');
Min20thResidual = -Max20thResidual;
TestingRaw = load('TestingCollisionDataRaw.csv');
%% Ground Truth
Motor_Torque = TestingRaw(:,32:33);
JTS = TestingRaw(:,36:37);
VelMFree = TestingRaw(:,6:7);
%% MOB
ResidualEstimate = TestingRaw(:,residual_idx:residual_idx+1);

%% LSTM
cd ..
cd result

FrictionModelLSTM = load('testing_result_collision.csv');
for i = 1:2
    FrictionModelLSTM(:,i) = (Max20thResidual(i) - Min20thResidual(i)) * FrictionModelLSTM(:,i)/2 + (Max20thResidual(i) + Min20thResidual(i))/2;
end
FrictionModelLSTM = [ResidualEstimate(1,:);FrictionModelLSTM]; % pandas does not read the first line
LSTMDataNum = size(FrictionModelLSTM,1);


%% Plot Trajectory
f1 = figure;
for j=1:2
    subplot(1,2,j)
    %plot(Motor_Torque(:,j) - JTS(:,j))
    plot(ResidualEstimate(:,j))
    hold on
    plot(FrictionModelLSTM(:,j))
    legend('MOB','LSTM')
end