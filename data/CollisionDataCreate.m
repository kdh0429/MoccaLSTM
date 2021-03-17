format long

%% For normalization
hz = 100;
num_data_type = 2;
num_joint = 2;
num_input = num_joint*num_data_type;
num_output = num_joint;
num_time_step = 100;

MaxTrainingData = load('MaxTrainingData.csv');
MinTrainingData = load('MinTrainingData.csv');

residual_idx = load('ResiIdx.csv');
MaxResidual = load('ResiMax.csv');
MinResidual = -MaxResidual;
%% Data Set Concatenate
Collision_Aggregate_Data = [];
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
        if strcmp(FolderName(k).name, 'collision')
            cd('collision')
            fileName = dir ('training_random_collision*.txt');
            for l = 1:size(fileName)
                Data = load(fileName(l).name);
                Collision_Aggregate_Data = vertcat(Collision_Aggregate_Data, Data);
            end
            disp(size(Collision_Aggregate_Data,1))
            cd ..;
        end
    end
    cd ..;
end
%% Data Preprocess
RawData= zeros(size(Collision_Aggregate_Data));
CollisionProcessData= zeros(size(Collision_Aggregate_Data,1), num_input*num_time_step+num_output);
CollisionProcessDataIdx = 1;
recent_wrong_dt_idx = 0;

for k=num_time_step+1:size(Collision_Aggregate_Data,1)
    % Check time stamp
    dt_data = round(Collision_Aggregate_Data(k,1) - Collision_Aggregate_Data(k-2,1),3);
    if dt_data ~= 2/hz
        recent_wrong_dt_idx = k;
    end
        
    if k < recent_wrong_dt_idx + num_time_step
        continue
    end
    
%     if norm(Collision_Aggregate_Data(k,30:31)) == 0
%         continue
%     end

    % Output
    for joint_data = 1:2
        CollisionProcessData(CollisionProcessDataIdx,num_input*num_time_step+joint_data) = 2*(Collision_Aggregate_Data(k,residual_idx-1+joint_data) - MinResidual(joint_data))/(MaxResidual(joint_data) - MinResidual(joint_data)) -1;
    end
    
    % Input
   for time_step=1:num_time_step
        corri_idx = 1;
        for joint_data=1:2
            CollisionProcessData(CollisionProcessDataIdx,num_input*(num_time_step-time_step)+joint_data) = 2*(Collision_Aggregate_Data(k-time_step+1,31+joint_data) - MinTrainingData(1,31+joint_data)) / (MaxTrainingData(1,31+joint_data) - MinTrainingData(1,31+joint_data)) -1; % theta
            CollisionProcessData(CollisionProcessDataIdx,num_input*(num_time_step-time_step)+2+joint_data) = 2*(Collision_Aggregate_Data(k-time_step+1,5+joint_data) - MinTrainingData(1,5+joint_data)) / (MaxTrainingData(1,5+joint_data) - MinTrainingData(1,5+joint_data)) -1; % theta dot
            %CollisionProcessData(CollisionProcessDataIdx,num_input*(num_time_step-time_step)+4+joint_data)= 2*(Collision_Aggregate_Data(k-time_step+1,1+joint_data) - MinTrainingData(1,1+joint_data)) / (MaxTrainingData(1,1+joint_data) -MinTrainingData(1,1+joint_data)) -1; %theta_dot_pre
            %CollisionProcessData(CollisionProcessDataIdx,num_input*(num_time_step-time_step)+6+joint_data)= 2*(Collision_Aggregate_Data(k-time_step,7+joint_data) - MinTrainingData(1,7+joint_data)) / (MaxTrainingData(1,7+joint_data) -MinTrainingData(1,7+joint_data)) -1; %theta_dot_desired
        end
   end
   
    RawData(CollisionProcessDataIdx,:) = Collision_Aggregate_Data(k,:);
    CollisionProcessDataIdx = CollisionProcessDataIdx +1;
end
CollisionProcessDataIdx = CollisionProcessDataIdx-1;

disp(CollisionProcessDataIdx)

csvwrite('TestingCollisionDataRaw.csv', RawData);
csvwrite('TestingCollisionDataFriction.csv', CollisionProcessData);