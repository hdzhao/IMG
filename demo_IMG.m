% Code for paper "Zhao, et al., Incomplete Multi-modal Visual Data Grouping, IJCAI'16"
% contact handong.zhao@gmail.com if you have any questions
% Oct. 22nd, 2016

clear;
addpath(genpath('measure/'));
addpath(genpath('misc/'));
datasetdir='data/';
num_views =2;

dataname={'buaa'};
numClust = 10;
nmi_All = [];


% Parameter to set partial/incomplete example ratio
% choose from [0.1 0.3 0.5 0.7 0.9]
pairPortion=[0.5];


% Parameters for the model
option.lamda=1e-2;
option.beta=1;
option.gamma=1e2;

for idata=1:length(dataname)
    dataf=strcat(datasetdir,dataname(idata),'RnSp.mat');
    datafname=cell2mat(dataf(1));
    load (datafname);
    if strcmp(dataname, 'buaa')
        dataV1 = dataV1(:,1:numClust*9);
        dataV2 = dataV2(:,1:numClust*9);
        dataV1 = NormalizeFea(dataV1,1);
        dataV2 = NormalizeFea(dataV2,1);
        X{1} =dataV1';
        X{2} =dataV2';
        truth = classLabel(1:numClust*9);
    else
        fprintf('No such data!');
    end
    
    load(cell2mat(strcat(datasetdir,dataname(idata),'Folds.mat'))); % folds
    [numFold,numInst]=size(folds);
   
    for f=1:numFold
        instanceIdx=folds(f,:);
        truthF=truth(instanceIdx);
        for v1=1:num_views
            for v2=v1+1:num_views
                if v1==v2
                    continue;
                end
                
                for pairedIdx=1:length(pairPortion) % different percentage of paired instances
                    numpairedInst=floor(numInst*pairPortion(pairedIdx));  % number of paired instances  have complete views
                    paired=instanceIdx(1:numpairedInst);
                    singledNumView1=ceil(0.5*(length(instanceIdx)-numpairedInst));
                    singleInstView1=instanceIdx(numpairedInst+1:numpairedInst+singledNumView1);   % the first view  and second view miss  half to half
                    singleInstView2=instanceIdx(numpairedInst+singledNumView1+1:end); %instanceIdx(numpairedInst+numsingleInstView1+1:end);
                    xpaired=X{v1}(paired,:);
                    ypaired=X{v2}(paired,:);
                    xsingle=X{v1}(singleInstView1,:);
                    ysingle=X{v2}(singleInstView2,:);
                    
                    option.latentdim=numClust;
                    
                    [U1 U2 P2 P1 P3 F P R nmi avgent AR] = IMGclust(xpaired,ypaired,xsingle,ysingle,numClust,truthF,option);
                    nmi_All = [nmi_All nmi];
                    
                end
            end
        end
    end
end



