function DrugResponse_main(inputfile)
%
% Build drug response model base on NCI60 data set
%
% Arguments:
% inputfile     Processed NCI60 data set (drug_mas5_reduced.mat) 
%               drugname     8 FDA proved drugs
%               genename     gene ID
%               marker       probe level ID
%               name         59 cell lines name
%               x            gene expression
%               y            GI50
% Output: 
% txt
%               Test stats
%               Train and test cell lines
%               Selected gene and weights
%               Cell line scores
% pdf
%               SVM prediction and stats of all cell lines
%
%
% Cai Huang 06/01/2016
%
    load(inputfile)
    [~,n_drug] = size(drugname);
    feature_left_fix = 1; %minimal number of selected genes
    factor=3/4; %separation factor toseparate training and testing data
    threshold = 0.5; %threshold for non determined data
    %%
    %iterate for all drugs
    for drug = 1:n_drug
        DrugResponse_model(inputfile,drug,feature_left_fix,factor,threshold);
    end
end