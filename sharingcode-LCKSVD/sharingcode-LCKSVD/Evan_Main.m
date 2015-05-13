clear all;
clc;
addpath('F:\Graduate Design\Database');
addpath('F:\Graduate Design\Database\ROI');
load('MCI403_ROI_5tpt');

[row,col,cell] = size(pMCI_data);
pMCI_1 = reshape(pMCI_data(:,1,:),[row,cell]);
pMCI_2 = reshape(pMCI_data(:,2,:),[row,cell]);
pMCI_3 = reshape(pMCI_data(:,3,:),[row,cell]);
pMCI_4 = reshape(pMCI_data(:,4,:),[row,cell]);
pMCI_5 = reshape(pMCI_data(:,5,:),[row,cell]);

[row,col,cell] = size(sMCI_data);
sMCI_1 = reshape(sMCI_data(:,1,:),[row,cell]);
sMCI_2 = reshape(sMCI_data(:,2,:),[row,cell]);
sMCI_3 = reshape(sMCI_data(:,3,:),[row,cell]);
sMCI_4 = reshape(sMCI_data(:,4,:),[row,cell]);
sMCI_5 = reshape(sMCI_data(:,5,:),[row,cell]);

datalabel = [ones(1,size(pMCI_1,2)),zeros(1,size(sMCI_1,2));
             zeros(1,size(pMCI_1,2)),ones(1,size(sMCI_1,2))];
DATALABEL = [ones(1,size(pMCI_1,2)),2.*ones(1,size(sMCI_1,2))];
data_1 = [pMCI_1,sMCI_1];
data_1 = data_1(1:4:size(data_1,1),:);
data_2 = [pMCI_2,sMCI_2];
data_2 = data_2(1:4:size(data_2,1),:);
data_3 = [pMCI_3,sMCI_3];
data_3 = data_3(1:4:size(data_3,1),:);
data_4 = [pMCI_4,sMCI_4];
data_4 = data_4(1:4:size(data_4,1),:);
data_5 = [pMCI_5,sMCI_5];
data_5 = data_5(1:4:size(data_5,1),:);

clear  row col cell 
clear  pMCI_1 pMCI_2 pMCI_3 pMCI_4 pMCI_5;
clear  sMCI_1 sMCI_2 sMCI_3 sMCI_4 sMCI_5;

c = cvpartition(DATALABEL,'k',10);
for k = 1:10
    
    Xt_1 = data_1(:,training(c,k));
    Lt_1 = datalabel(:,training(c,k));
    Xt_2 = data_2(:,training(c,k));
    Lt_2 = datalabel(:,training(c,k));
    Xt_3 = data_3(:,training(c,k));
    Lt_3 = datalabel(:,training(c,k));
    Xt_4 = data_4(:,training(c,k));
    Lt_4 = datalabel(:,training(c,k));
    Xt_5 = data_5(:,training(c,k));
    Lt_5 = datalabel(:,training(c,k));

    Xs_1 = data_1(:,test(c,k));
    Ls_1 = datalabel(:,test(c,k));
    Xs_2 = data_2(:,test(c,k));
    Ls_2 = datalabel(:,test(c,k));
    Xs_3 = data_3(:,test(c,k));
    Ls_3 = datalabel(:,test(c,k));
    Xs_4 = data_4(:,test(c,k));
    Ls_4 = datalabel(:,test(c,k));
    Xs_5 = data_5(:,test(c,k));
    Ls_5 = datalabel(:,test(c,k));

    Xt = zeros(size(Xt_1,1),5*size(Xt_1,2));
    Xt(:,1:5:5*(size(Xt_1,2)-1)+1) = Xt_1;
    Xt(:,2:5:5*(size(Xt_1,2)-1)+2) = Xt_2;
    Xt(:,3:5:5*(size(Xt_1,2)-1)+3) = Xt_3;
    Xt(:,4:5:5*(size(Xt_1,2)-1)+4) = Xt_4;
    Xt(:,5:5:5*(size(Xt_1,2))) = Xt_5;

    Xs = zeros(size(Xs_1,1),5*size(Xs_1,2));
    Xs(:,1:5:5*(size(Xs_1,2)-1)+1) = Xs_1;
    Xs(:,2:5:5*(size(Xs_1,2)-1)+2) = Xs_2;
    Xs(:,3:5:5*(size(Xs_1,2)-1)+3) = Xs_3;
    Xs(:,4:5:5*(size(Xs_1,2)-1)+4) = Xs_4;
    Xs(:,5:5:5*(size(Xs_1,2))) = Xs_5;

    Lt = zeros(size(Lt_1,1),5*size(Lt_1,2));
    Lt(:,1:5:5*(size(Lt_1,2)-1)+1) = Lt_1;
    Lt(:,2:5:5*(size(Lt_1,2)-1)+2) = Lt_2;
    Lt(:,3:5:5*(size(Lt_1,2)-1)+3) = Lt_3;
    Lt(:,4:5:5*(size(Lt_1,2)-1)+4) = Lt_4;
    Lt(:,5:5:5*(size(Lt_1,2))) = Lt_5;

    Ls = zeros(size(Ls_1,1),5*size(Ls_1,2));
    Ls(:,1:5:5*(size(Ls_1,2)-1)+1) = Ls_1;
    Ls(:,2:5:5*(size(Ls_1,2)-1)+2) = Ls_2;
    Ls(:,3:5:5*(size(Ls_1,2)-1)+3) = Ls_3;
    Ls(:,4:5:5*(size(Ls_1,2)-1)+4) = Ls_4;
    Ls(:,5:5:5*(size(Ls_1,2))) = Ls_5;
    
    %% constant
    sparsitythres = 30; % sparsity prior
    sqrt_alpha = 4; % weights for label constraint term
    sqrt_beta = 2; % weights for classification err term
    dictsize = 570; % dictionary size
    iterations = 50; % iteration number
    iterations4ini = 20; % iteration number for initialization
    
    %% dictionary learning process
    % get initial dictionary Dinit and Winit
    fprintf('\nLC-KSVD initialization... ');
    [Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(Xt,Lt,dictsize,iterations4ini,sparsitythres);
    fprintf('done!');
    
    % run LC k-svd training (reconstruction err + class penalty + classifier err)
    fprintf('\nDictionary learning by LC-KSVD1...');
    [D1,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
    save('.\trainingdata\dictionarydata1.mat','D1','X1','W1','T1');
    fprintf('done!');
    
%     fprintf('\nDictionary and classifier learning by LC-KSVD2...')
%     [D2,X2,T2,W2] = labelconsistentksvd2(Xt,Dinit,Q_train,Tinit,Lt,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
%     save('.\trainingdata\dictionarydata2.mat','D2','X2','W2','T2');
%     fprintf('done!');
    
    [prediction1,accuracy1] = classification(D1, W1, testing_feats, H_test, sparsitythres);
    fprintf('\nFinal recognition rate for LC-KSVD1 is : %.03f ', accuracy1);

%     [prediction2,accuracy2] = classification(D2, W2, Xs, Ls, sparsitythres);
%     fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);
end
    
    
    
