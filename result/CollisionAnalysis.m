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

%%
threshold = 1.0*load('Threshold.csv');
dt = 0.01;
last_t = 0.0;
t= last_t:dt:last_t+(size(FrictionModelLSTM,1)-1)*dt;
last_t = last_t + size(FrictionModelLSTM,1)*dt;
DOB_Free = ResidualEstimate(1:size(FrictionModelLSTM,1),:) - FrictionModelLSTM;
DOB_Detection_joint = [];
DOB_Detection_time = [];
DOB_Detection = 0;
for i=1:size(FrictionModelLSTM,1)
    if (abs(DOB_Free(i,1))>threshold(1) || abs(DOB_Free(i,2))>threshold(2))
        continueous_col = 0;
        DOB_Detection = DOB_Detection +1;
        DOB_Detection_time(DOB_Detection) = t(i);
        for joint = 1:2
            if abs(DOB_Free(i,joint))>threshold(joint)
                DOB_Detection_joint(DOB_Detection) = joint;
            end
        end
    end
end

disp("-----------------------------")
disp("FP DoB:")
disp(DOB_Detection)
disp("-----------------------------")
disp("DOB Detection Time:")
fprintf('Joint %d\n',DOB_Detection_joint(1));
for i=2:DOB_Detection
    del_time = abs(DOB_Detection_time(i)-DOB_Detection_time(i-1));
    if( del_time> 0.5)
        disp(del_time)
        fprintf('Joint %d\n',DOB_Detection_joint(i));
    end
end

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