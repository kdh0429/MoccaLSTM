clear all
clc
format long

%% For normalization
hz = 100;
num_data_type = 2;
num_joint = 2;
num_input = num_joint*num_data_type;
num_output = num_joint;
num_time_step = 100;

residual_type = 4; % 0: MOB no error, 1: MOB inertia error, 2: MOB mass error, 3: MOB com pos error, 4: MOB inertia, mass, com pos error, 5: MOB without model
residual_idx = 38;
if residual_type == 0
    residual_idx = 38;
elseif residual_type == 1
    residual_idx = 44;
elseif residual_type == 2
    residual_idx = 50;
elseif residual_type == 3
    residual_idx = 56;
elseif residual_type == 4
    residual_idx = 62;
elseif residual_type == 5
    residual_idx = 68;
end

MaxTrainingData = ...
[1810, ...                              % 1     time
6.28316000000000,6.28315000000000,  ... % 2     theta [0, 360)
8.90825000000000,9.57301000000000,  ... % 4     theta
2.07162000000000,2.16591000000000,  ... % 6     theta dot 
0,0,                                ... % 8     theta dot pre
2.06906000000000,2.16790000000000,  ... % 10    theta dot lpf
2.06906000000000,2.16790000000000,  ... % 12    theta dot pre lpf
2068.12000000000,2165.50000000000,  ... % 14    theta double dot lpf
6.28302000000000,6.28316000000000,  ... % 16    q [0, 360)
5718.37000000000,9612.38000000000,  ... % 18    q dot
0,0,                                ... % 20    q dot lpf
0,0,                                ... % 22    q dot pre lpf
0,0,                                ... % 24    q double dot lpf
0,0,                                ... % 26    q double dot pre lpf
8.92087000000000,9.58078000000000,  ... % 28    q desired
1.96008000000000,1.96326000000000,  ... % 30    q dot desired
90.5760000000000,32.8320000000000,  ... % 32    motor torque
90.0103000000000,32.6757000000000,  ... % 34    command torque
65.0940000000000,2.11182000000000,  ... % 36    JTS
24.2791000000000,13.9615000000000,25.0961000000000,15.1633000000000,26.4307000000000,16.9608000000000, ... % 38    MOB no error
24.3076000000000,13.9364000000000,25.2531000000000,15.0315000000000,26.5200000000000,16.6877000000000, ... % 44    MOB inertia error
52.3266000000000,21.6903000000000,53.3539000000000,22.8187000000000,54.4899000000000,24.5044000000000, ... % 50    MOB mass error
37.4248000000000,21.6748000000000,38.7810000000000,22.7376000000000,39.5806000000000,24.3361000000000, ... % 56    MOB com pos error
58.9280000000000,25.5219000000000,61.0438000000000,26.4741000000000,62.1361000000000,27.9188000000000, ... % 62    MOB inertia, mass, com pos error
81.0565000000000,29.3689000000000,83.7459000000000,30.2106000000000,84.6915000000000,31.5016000000000];    % 68    MOB without model

MinTrainingData = ...
[6.01000000000000, ...                    % 1     time
-3.01134000000000,-3.49399000000000,  ... % 2     theta [0, 360)
-3.01134000000000,-3.49399000000000,  ... % 4     theta
-2.01426000000000,-2.36291000000000,  ... % 6     theta dot 
0,0,                                  ... % 8     theta dot pre
-2.01345000000000,-2.36390000000000,  ... % 10    theta dot lpf
-2.01345000000000,-2.36390000000000,  ... % 12    theta dot pre lpf
-2009.87000000000,-2358.25000000000,  ... % 14    theta double dot lpf
-3.01219000000000,-3.49314000000000,  ... % 16    q [0, 360)
-6203.24000000000,-3454.20000000000,  ... % 18    q dot
0,0,                                  ... % 20    q dot lpf
0,0,                                  ... % 22    q dot pre lpf
0,0,                                  ... % 24    q double dot lpf
0,0,                                  ... % 26    q double dot pre lpf
-3.01873000000000,-3.49838000000000,  ... % 28    q desired
-1.96063000000000,-1.96318000000000,  ... % 30    q dot desired
-93.0240000000000,-32.1120000000000,  ... % 32    motor torque
-92.0971000000000,-31.4786000000000,  ... % 34    command torque
-64.4958000000000,-1.81274000000000,  ... % 36    JTS
-23.9739000000000,-13.3314000000000,-25.4423000000000,-13.9007000000000,-26.4318000000000,-15.2902000000000, ... % 38    MOB no error
-24.0413000000000,-13.3042000000000,-25.4179000000000,-13.8488000000000,-26.5843000000000,-15.1048000000000, ... % 44    MOB inertia error
-52.1566000000000,-20.8849000000000,-54.1288000000000,-21.3752000000000,-55.2531000000000,-22.1474000000000, ... % 50    MOB mass error
-37.3865000000000,-20.8747000000000,-39.1834000000000,-21.3432000000000,-40.3346000000000,-22.0539000000000, ... % 56    MOB com pos error
-59.3925000000000,-24.6553000000000,-61.5040000000000,-25.0828000000000,-62.5408000000000,-25.7415000000000, ... % 62    MOB inertia, mass, com pos error
-81.4351000000000,-28.4360000000000,-83.8375000000000,-28.8301000000000,-84.7470000000000,-29.4291000000000];    % 68    MOB without model

%% Data Set Concatenate
Free_Aggregate_Data = [];
FolderName = dir;
folder_idx = 1;
for time_step = 1:size(FolderName,1)
    if ((size(FolderName(time_step).name,2) > 4) && (strcmp(FolderName(time_step).name(1:4), '2020')))
        DataFolderList(folder_idx) = string(FolderName(time_step).name);
        folder_idx = folder_idx + 1;
    end
end

% 폴더별 자유모션 데이터 합치기 
for joint_data = 1:size(DataFolderList,2)
    cd (DataFolderList(joint_data))
    FolderName = dir;
    for k = 1:size(FolderName,1)
        if strcmp(FolderName(k).name, 'free')
            cd('free')
            fileName = dir ('training_random_free*.txt');
            for l = 1:size(fileName)
                Data = load(fileName(l).name);
                Free_Aggregate_Data = vertcat(Free_Aggregate_Data, Data);
            end
            disp(size(Free_Aggregate_Data,1))
            cd ..;
        end
    end
    cd ..;
end

MaxTrainingData = max(Free_Aggregate_Data,[],1);
MinTrainingData = - MaxTrainingData;
MaxResidual = MaxTrainingData(residual_idx:residual_idx+1);
MinResidual = -MaxResidual;
csvwrite('ResiMax.csv', MaxResidual);
csvwrite('ResiIdx.csv', residual_idx);
csvwrite('MaxTrainingData.csv', MaxTrainingData);
csvwrite('MinTrainingData.csv', MinTrainingData);
%% Data Preprocess
RawData= zeros(size(Free_Aggregate_Data));
FreeProcessData= zeros(size(Free_Aggregate_Data,1), num_input*num_time_step+num_output);
FreeProcessDataIdx = 1;
recent_wrong_dt_idx = 0;

for k=num_time_step+1:size(Free_Aggregate_Data,1)
    % Check time stamp
    dt_data = round(Free_Aggregate_Data(k,1) - Free_Aggregate_Data(k-2,1),3);
    if dt_data ~= 2/hz
        recent_wrong_dt_idx = k;
    end
        
    if k < recent_wrong_dt_idx + num_time_step
        continue
    end
    
%     if norm(Free_Aggregate_Data(k,30:31)) == 0
%         continue
%     end

    % Output
    for joint_data = 1:2
        FreeProcessData(FreeProcessDataIdx,num_input*num_time_step+joint_data) = 2*(Free_Aggregate_Data(k,residual_idx-1+joint_data) - MinResidual(joint_data))/(MaxResidual(joint_data) - MinResidual(joint_data)) -1;
    end
    
    % Input
   for time_step=1:num_time_step
        corri_idx = 1;
        for joint_data=1:2
            FreeProcessData(FreeProcessDataIdx,num_input*(num_time_step-time_step)+joint_data) = 2*(Free_Aggregate_Data(k-time_step+1,31+joint_data) - MinTrainingData(1,31+joint_data)) / (MaxTrainingData(1,31+joint_data) - MinTrainingData(1,31+joint_data)) -1; % theta
            FreeProcessData(FreeProcessDataIdx,num_input*(num_time_step-time_step)+2+joint_data) = 2*(Free_Aggregate_Data(k-time_step+1,5+joint_data) - MinTrainingData(1,5+joint_data)) / (MaxTrainingData(1,5+joint_data) - MinTrainingData(1,5+joint_data)) -1; % theta dot
            %FreeProcessData(FreeProcessDataIdx,num_input*(num_time_step-time_step)+4+joint_data)= 2*(Free_Aggregate_Data(k-time_step+1,1+joint_data) - MinTrainingData(1,1+joint_data)) / (MaxTrainingData(1,1+joint_data) -MinTrainingData(1,1+joint_data)) -1; %theta_dot_pre
            %FreeProcessData(FreeProcessDataIdx,num_input*(num_time_step-time_step)+6+joint_data)= 2*(Free_Aggregate_Data(k-time_step+1,7+joint_data) - MinTrainingData(1,7+joint_data)) / (MaxTrainingData(1,7+joint_data) -MinTrainingData(1,7+joint_data)) -1; %theta_dot_pre
        end
   end
   
    RawData(FreeProcessDataIdx,:) = Free_Aggregate_Data(k,:);
    FreeProcessDataIdx = FreeProcessDataIdx +1;
end
FreeProcessDataIdx = FreeProcessDataIdx-1;

disp(FreeProcessDataIdx)

TrainingRaw = RawData(fix(0.2*FreeProcessDataIdx):fix(1.0*FreeProcessDataIdx),:);
TestingRaw = RawData(1:fix(0.1*FreeProcessDataIdx),:);

TrainingData = FreeProcessData(fix(0.2*FreeProcessDataIdx):fix(1.0*FreeProcessDataIdx),:);
ValidationData = FreeProcessData(fix(0.1*FreeProcessDataIdx):fix(0.2*FreeProcessDataIdx),:);
TestingData = FreeProcessData(1:fix(0.1*FreeProcessDataIdx),:);
clear FreeProcessData;

TrainingDataMix = TrainingData(randperm(size(TrainingData,1)),:);
clear TrainingData;

csvwrite('TrainingDataRaw.csv', TrainingRaw);
csvwrite('TestingDataRaw.csv', TestingRaw);
csvwrite('TrainingDataFriction.csv', TrainingDataMix);
csvwrite('ValidationDataFriction.csv', ValidationData);
csvwrite('TestingDataFriction.csv', TestingData);

%%
CollisionDataCreate