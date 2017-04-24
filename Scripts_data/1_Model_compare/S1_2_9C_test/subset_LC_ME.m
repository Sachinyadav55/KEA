close all
clear
clc
%%
% mas5 data
load drug_mas5_reduced_LC_ME.mat
%%

%%
[~,n_drug] = size(drugname);
feature_left_fix = 10;
factor=3/4; %separation factor toseparate training and testing data
threshold = 0.5; %threshold for non determined data
l18_acc_ak=[];
drug = 2% drug Doxorubicin
    drugname{drug} %print drug name
%%
    %get label and remove non determined data
    y_drug = y(:,drug); good_idx = ~isnan(y_drug);
    name = name(good_idx);
    x = x(good_idx,:);
    y_drug = y_drug(good_idx);
    mean_y = mean(y_drug);
    std_y = std(y_drug);
    z_drug = (y_drug-mean_y)/std_y;
%%
    %select data and separate training and testing data
    select_idx = abs(z_drug) > threshold;
    x_select = x(select_idx,:);
    y_select = y_drug(select_idx);
    z_select = z_drug(select_idx);
    label_select = z_select > 0;
    name_select = name(select_idx);
    pos_z = find(z_select > 0); [~,pos_I] = sort( z_select(pos_z) );
    neg_z = find(z_select < 0); [~,neg_I] = sort( -z_select(neg_z) );
    train_idx = [pos_z(pos_I(1:floor(length(pos_I)*factor))); neg_z(neg_I(1:floor(length(neg_I)*factor)))];
    test_idx = setdiff([1:length(z_select)]', train_idx);
    remain = 1:size(x,2);
    sorted_feature = [];
 %%     
    %recersive calculate weights for features, and remove features base on the weights
    n_feature = 0; step = 100;
    while n_feature < size(x,2)
        svm = svmtrain(x_select(train_idx,remain), label_select(train_idx),'kernel_function','linear','boxconstraint',1,'method','SMO');
        w = svm.Alpha'*svm.SupportVectors;
        if (length(remain) > step) 
            [~,I]=sort(abs(w)); I = I(step:-1:1); % step =100
        else
            [~,I]=min(abs(w)); I = I(1); % step = 1
        end

        next_feature = remain(I);
        sorted_feature = [next_feature sorted_feature];
        remain(I) = [];

        n_feature = length(sorted_feature);
        if (n_feature == size(x,2)) || (mod(n_feature,step) == 0)
            fprintf('%d ', n_feature);
            if mod(n_feature/step,10) == 0, fprintf('\n'); end
        end
    end
    fprintf('\n');
%% 
    %test the features left on testing data
    acc = [];
    for f = [1:step] %[1:step-1 step:step:size(x,2) size(x,2)]
        xtrain = x_select(train_idx,sorted_feature(1:f)); ytrain = label_select(train_idx);
        xtest = x_select(test_idx,sorted_feature(1:f)); ytest = label_select(test_idx);
        cp = classperf(label_select,'Positive',1,'Negative',0);
        svm = svmtrain(xtrain, ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
        classOut = svmclassify(svm,xtest);
        cp = classperf(cp,classOut,test_idx);
        acc = [acc [cp.CorrectRate cp.Sensitivity cp.Specificity]'];

        if (f == size(x,2)) || (mod(f,1) == 0)
            fprintf('%d ', f);
            if mod(f/1,10) == 0, fprintf('\n'); end
        end
    end
    fprintf('\n');
%%  
    %make sure number seleted features is greater than threshold
    [~,I] = max(acc(1,:)); 
    if I(1)<feature_left_fix
        I=feature_left_fix;
    else
        I=I(1);
    end
      
    fprintf('%f %f %f\n', acc(:,I));
    max_feature = sorted_feature(1:I); accuracy = acc(1,I)*100; sensitivity = acc(2,I)*100; specificity = acc(3,I)*100;
    feature_name = marker(max_feature)
    gene_name = genename(max_feature)

    %delete(gcp('nocreate'))
    n_worker = 20;
    %myCluster=parcluster('local'); myCluster.NumWorkers=n_worker; parpool(myCluster,n_worker);
%%
    % permutation test
    xtrain = x_select(train_idx,max_feature); ytrain = label_select(train_idx);
    xtest = x_select(test_idx,max_feature); ytest = label_select(test_idx);
    N = 1000;
    count = zeros(1,N);
    for i=1:N
        p = randperm(size(xtrain,1)); yperm = ytrain(p);
        cp = classperf(label_select,'Positive',1,'Negative',0);
        svm = svmtrain(xtrain, yperm,'kernel_function','linear','boxconstraint',1,'method','SMO');
        classOut = svmclassify(svm,xtest);
        cp = classperf(cp,classOut,test_idx);
        if cp.CorrectRate*100 >= accuracy, count(i) = 1; end
        %if mod(i,100) == 0, fprintf('%d %.4f ', i, count/N); end
    end
    p_value = sum(count) / N;
    fprintf('\np_value=%.4f\n',p_value);
    %delete(gcp('nocreate'))
    %clear myCluster
%%
    %Leave One Out
    lsa=[];
    cp = classperf(label_select,'Positive',1,'Negative',0);
    for i=1:size(x_select,1)
        LOO_test_idx = false(size(x_select,1),1); LOO_test_idx(i) = true;
        LOO_train_idx = ~LOO_test_idx;
        LOO_xtrain = x_select(LOO_train_idx,max_feature); LOO_ytrain=label_select(LOO_train_idx);
        LOO_xtest = x_select(LOO_test_idx,max_feature); LOO_ytest=label_select(LOO_test_idx);
        svm = svmtrain(LOO_xtrain, LOO_ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
        classOut = svmclassify(svm,LOO_xtest);
        cp = classperf(cp,classOut,LOO_test_idx);
        loow = svm.Alpha'*svm.SupportVectors;
        loow_noscale = loow .* svm.ScaleData.scaleFactor;
        loob_noscale = sum(loow .* svm.ScaleData.scaleFactor .* svm.ScaleData.shift) + svm.Bias;
        looscore = LOO_xtest*loow_noscale' + loob_noscale; looscore = -looscore;
        lsa=[lsa,looscore];
    end
    LOO_acc = [cp.CorrectRate cp.Sensitivity cp.Specificity] * 100;
    fprintf('\nLOO %.2f%%\n',LOO_acc);

    X = x_select(:,max_feature); Y = label_select;
    xtrain = x_select(train_idx,max_feature); ytrain = label_select(train_idx);
    svm = svmtrain(xtrain, ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
    w = svm.Alpha'*svm.SupportVectors;
    XX = bsxfun(@plus, X, svm.ScaleData.shift); XX = bsxfun(@times, XX, svm.ScaleData.scaleFactor);
    (XX*w'+svm.Bias)' < 0 == Y'
    w_noscale = w .* svm.ScaleData.scaleFactor;
    b_noscale = sum(w .* svm.ScaleData.scaleFactor .* svm.ScaleData.shift) + svm.Bias;
    (X*w_noscale'+b_noscale)' < 0 == Y'
    assert( all((X*w_noscale'+b_noscale < 0) == (XX*w'+svm.Bias < 0) ) )
    score = x(:,max_feature)*w_noscale' + b_noscale; score = -score;
    pos_idx = z_drug > threshold; neg_idx = z_drug < -threshold; not_idx = ~pos_idx & ~neg_idx;
    tmp = find(select_idx);
    train_mark = false(size(x,1),1); train_mark(tmp(train_idx)) = true;
    thry = 0.5*std_y;
%%    
    %print out all data points
    output_file = ['svm4_' drugname{drug} 'lc_me'];
    clf, hold on
    h(1)=plot( score(pos_idx & train_mark), y_drug(pos_idx & train_mark) , 'or','MarkerSize',18,'LineWidth',3);
    h(2)=plot( score(neg_idx & train_mark), y_drug(neg_idx & train_mark) , 'ob','MarkerSize',18,'LineWidth',3 );
    h(3)=plot( score(~train_mark), y_drug(~train_mark) , 'dk','MarkerSize',18,'LineWidth',3);
    h(4)=plot( score(not_idx), y_drug(not_idx) , 'dk', 'MarkerSize',18,'LineWidth',3);
    h(5)=plot( [min(score(select_idx))-.3 max(score(select_idx))+.3],[mean(y_drug)-thry, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(6)=plot( [min(score(select_idx))-.3 max(score(select_idx))+.3],[mean(y_drug)+thry, mean(y_drug)+thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    set(h(1), 'MarkerFaceColor', 'r');
    set(h(2), 'MarkerFaceColor', 'b');
    t(1)=title(sprintf('%s\nacc=%.2f%% sn=%.2f%% sp=%.2f%%', drugname{drug}, LOO_acc));
    t(2)=xlabel( 'score');
    t(3)=ylabel( 'log transformed GI50');
    set(t, 'FontSize',40);
    set(gca,'FontSize',20);
    xlim([min(score(select_idx))-.3, max(score(select_idx))+.3]);
    ylim([min(y_drug(select_idx))-.3, max(y_drug(select_idx))+.3]);
    legend('Sensitive (Train)', 'Resistant (Train)', 'Test' , 'Location','bestoutside');
    
    h_vert = line([0 0], ylim);
    set(h_vert, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 2);
    h_horiz = line(xlim, [1 1]*mean(y_drug));
    set(h_horiz, 'LineStyle', '--', 'Color', 'k', 'LineWidth', 2);
    lx = xlim; ly = ylim;
    th(1) = text(lx(1), ly(1), 'True Negative', 'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Bottom');
    th(2) = text(lx(1), ly(2), 'False Negative', 'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Top');
    th(3) = text(lx(2), ly(2), 'True Positive', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');
    th(4) = text(lx(2), ly(1), 'False Positive', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
    set(th, 'FontSize', 20);
    print_figure(gcf,[15 10], output_file,'-dpdf')
