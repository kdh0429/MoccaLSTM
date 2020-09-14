clear all;
clc
format long
%%
cd ../data/
residual_idx = load('ResiIdx.csv');
Max20thResidual = load('ResiMax.csv');
Min20thResidual = -Max20thResidual;
TestingRaw = load('TestingDataRaw.csv');
%% Ground Truth
Motor_Torque = TestingRaw(:,32:33);
JTS = TestingRaw(:,36:37);
VelMFree = TestingRaw(:,6:7);
%% MOB
ResidualEstimate = TestingRaw(:,residual_idx:residual_idx+1);

%% Evaluation with Frction Model
TrainingRaw = load('TrainingDataRaw.csv');
Motor_TorqueTraining = TrainingRaw(:,32:33);
VelMFreeTraining = TrainingRaw(:,6:7);
ResidualEstimateTraining = TrainingRaw(:,residual_idx:residual_idx+1);
x = abs(VelMFreeTraining);
y=abs(ResidualEstimateTraining);
p = zeros(4,2);
for j=1:2
    p(:,j) = polyfit(x(:,j),y(:,j),3);
end

FrictionModelPoly = zeros(size(VelMFree));
for j=1:2
    FrictionModelPoly(:,j) = sign(VelMFree(:,j)).*polyval(p(:,j),abs(VelMFree(:,j)));
end

%% LSTM
cd ..
cd result

FrictionModelLSTM = load('testing_result.csv');
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

%% Plot qdot
f2 = figure;
qdot = [0:0.001:2.1]';

for i=1:2
    fm(:,i) = polyval(p(:,i),qdot);
end

for j= 1:2
    subplot(1,2,j)
    plot(abs(VelMFree(1:LSTMDataNum,j)), abs(ResidualEstimate(1:LSTMDataNum,j)))
    hold on
    plot(abs(VelMFree(1:LSTMDataNum,j)), abs(FrictionModelLSTM(:,j)))
    plot(qdot(:),fm(:,j))
    legend('MOB','LSTM','Poly')
end

%% Plot Error histogram
Residual = ResidualEstimate(1:LSTMDataNum,:);
PolyErr = Residual(1:LSTMDataNum,:)-FrictionModelPoly(1:LSTMDataNum,:);
LSTMErr = Residual(1:LSTMDataNum,:)-FrictionModelLSTM;
f3 = figure;

% num_bin = 100;
% bin_max = [20,30,15,6,5,10];
% bin_min = [-20,-30,-15,-6,-5,-10];
% bin_arr = [bin_min(1):(bin_max(1)-bin_min(1))/num_bin:bin_max(1); ...
%     bin_min(2):(bin_max(2)-bin_min(2))/num_bin:bin_max(2); ...
%     bin_min(3):(bin_max(3)-bin_min(3))/num_bin:bin_max(3); ...
%     bin_min(4):(bin_max(4)-bin_min(4))/num_bin:bin_max(4); ...
%     bin_min(5):(bin_max(5)-bin_min(5))/num_bin:bin_max(5); ...
%     bin_min(6):(bin_max(6)-bin_min(6))/num_bin:bin_max(6)];
% 
% for j=1:2
%     subplot(1,2,j)
%     histogram(PolyErr(:,j),bin_arr(j,:))
%     hold on
%     histogram(LSTMErr(:,j),bin_arr(j,:))
%     legend('Poly', 'LSTM')
% end
for j=1:2
    subplot(1,2,j)
    histogram(PolyErr(:,j))
    hold on
    histogram(LSTMErr(:,j))
    legend('Poly', 'LSTM')
end

disp('Poly Error Mean: ')
mean(abs(PolyErr),1)
disp('LSTM Error Mean: ')
mean(abs(LSTMErr),1)
disp('LSTM Threshold: ')
[threshold, idx] = max(abs(LSTMErr),[],1)
csvwrite('Threshold.csv', threshold);
