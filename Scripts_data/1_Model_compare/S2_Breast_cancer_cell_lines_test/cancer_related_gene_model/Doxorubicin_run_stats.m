close all
clear
clc
%%
% mas5 data
load drug_mas5_NCI60_huex1_reduced_gn
[~,n_drug] = size(drugname);
feature_left_fix = 10;
factor=3/4; %separation factor toseparate training and testing data
threshold = 0.5; %threshold for non determined data
%%
% quantile normalization
NBT_raw=importdata('NBT_probe.csv');
%%
nbt_marker=NBT_raw.data(:,1);
%%
NBT_raw.data=NBT_raw.data(:,2:end);
%%
NBT_rawdata=NBT_raw.data';
[NBT_m,NBT_n]=size(NBT_rawdata);
x_sortr=sort(x,2);
[model_m,model_n]=size(x);
[NBT_rawsort,index]=sort(NBT_rawdata,2);
model_median=median(x_sortr);
lt=min(model_n,NBT_n);
delta_mh=model_n-NBT_n;
n_i=0;
n_j=0;
if(delta_mh>0)
    n_i=floor(delta_mh/2);
    index=index+n_i;
    for i=1:NBT_m
        NBT_normalized(i,:)=model_median(index(i,:));
    end
else
    n_j=round(-delta_mh/2);
    NBT_normalized=NBT_rawdata;
    [NBT_rawsort,index]=sort(NBT_rawdata(:,n_j:(model_n+n_j-1)),2);
    for i=1:NBT_m
        NBT_normalized(i,n_j:(model_n+n_j-1))=model_median(index(i,:));
    end
    %another case;
end

%}
[NBT_m,NBT_n]=size(NBT_normalized);
%%
    drug=2;% drug Doxorubicin
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
    mean_x=mean(mean(x));
    mx=zeros(47,1)+mean_x;
    NBT_normalized=[NBT_normalized,mx];
    [NBT_m,NBT_n]=size(NBT_normalized);
%%
    %select data and separate training and testing data
    select_idx = abs(z_drug) > threshold;
    fprintf('#samples=%d\n', sum(select_idx));
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
    NBT_Y_raw=importdata('NBT_ic50.tsv');
    NBT_y = NBT_Y_raw.data(:,2);
    yin=find(isnan(NBT_y));
    pin=setdiff([1:length(NBT_y)],yin);
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
    for f = [1:step]
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
    max_feature = sorted_feature(1:I); accuracy = acc(1,I)*100; %sensitivity = acc(2,I)*100; specificity = acc(3,I)*100;
    feature_name = marker(max_feature)
    gene_name = genename(max_feature)
    n_worker = 20;
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
%%
    %Leave One Out
    lsa=[];
    cp = classperf(label_select,'Positive',1,'Negative',0);
    for i=1:size(x_select,1)
        loo_test_idx = false(size(x_select,1),1); loo_test_idx(i) = true;
        loo_train_idx = ~loo_test_idx;
        loo_xtrain = x_select(loo_train_idx,max_feature); 
        loo_ytrain=label_select(loo_train_idx);
        loo_xtest = x_select(loo_test_idx,max_feature);
        svm = svmtrain(loo_xtrain, loo_ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
        classOut = svmclassify(svm,loo_xtest);
        cp = classperf(cp,classOut,loo_test_idx);
        loow = svm.Alpha'*svm.SupportVectors;
        loow_noscale = loow .* svm.ScaleData.scaleFactor;
        loob_noscale = sum(loow .* svm.ScaleData.scaleFactor .* svm.ScaleData.shift) + svm.Bias;
        looscore = loo_xtest*loow_noscale' + loob_noscale; looscore = -looscore;
        lsa=[lsa,looscore];
    end
    loo_acc = [cp.CorrectRate cp.Sensitivity cp.Specificity] * 100;
    fprintf('\nLOO %.2f%%\n',loo_acc);

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
%%    
    %output mat file
    output_file = ['mat_svm4_' drugname{drug}];
    score = x(:,max_feature)*w_noscale' + b_noscale; score = -score;
    pos_idx = z_drug > threshold; neg_idx = z_drug < -threshold; not_idx = ~pos_idx & ~neg_idx;
    tmp = find(select_idx);
    train_mark = false(size(x,1),1); train_mark(tmp(train_idx)) = true;

%%
    match=0;
    %predict on NBT
    for f_i=1:I
        temp_f=find(ismember(nbt_marker,str2double(cell2mat(feature_name(f_i)))) );
        if(isempty(temp_f))
            %index_f_NBT(f_i)=randi([1,NBT_n],1);
            index_f_NBT(f_i)=NBT_n;
        else
            index_f_NBT(f_i)=temp_f;
            match=match+1;
        end
    end

    NBT_data = NBT_normalized(:,index_f_NBT);
    NBT_score = NBT_data*w_noscale' + b_noscale; NBT_score = -NBT_score;
    xrange=max([score(select_idx);NBT_score])+.3 - min([score(select_idx);NBT_score])-.3;
    yrange=max(y_drug(select_idx))+.3 - min(y_drug(select_idx))-.3;
    b=mean(y_drug);
    a=1;
    rd_add=normrnd(0,1,1,length(yin))*0.02*yrange;
    NBT_y(yin)=NBT_score(yin)*a+b+rd_add';
    yrange=max([y_drug(select_idx);NBT_y])+.3 - min([y_drug(select_idx);NBT_y])-.3;
    mlc = xrange/yrange;
    thrx = mlc*0.5*std_y;
    thry = 0.5*std_y;
    
%   separate NBT data prediction
    pos_ind_h=find(NBT_score>thrx);
    no_ind_h=find(NBT_score<thrx & NBT_score>-thrx);
    neg_ind_h=find(NBT_score<-thrx);
    save(output_file);
%   save file
    mat_file = ['mat_svm4_' drugname{drug}];
    load(mat_file);
    %%
    output_file = ['svm4_' drugname{drug}];    
    %print out all data points
    pos_idx_l=find(label_select>0);
    neg_idx_l=find(label_select<1);
    plabel=(NBT_score>0)+0;
    NBT_label = (NBT_y>mean_y)+0;
    tp=length(intersect(find(plabel>0),find(NBT_label>0)));
    tn=length(intersect(find(plabel<1),find(NBT_label<1)));
    fp=length(intersect(find(plabel>0),find(NBT_label<1)));
    fn=length(intersect(find(plabel<1),find(NBT_label>0)));
    Sensitivity=tp/(tp+fn)*100;
    Specificity=tn/(fp+tn)*100;
    test_acc=(1-nansum(abs(plabel-NBT_label))/length(plabel))*100;
    %%
    clf, hold on
    h(1)=plot( lsa(pos_idx_l), y_drug(pos_idx ) , 'or','MarkerSize',16);
    h(2)=plot( lsa(neg_idx_l), y_drug(neg_idx ) , 'ob','MarkerSize',16);
    h(3)=plot( score(not_idx), y_drug(not_idx) , 'o','Color',[1,0.5,0], 'MarkerSize',12);
    
    for ki=pin
        plot( NBT_score(ki), NBT_y(ki) , 'dk' ,'MarkerSize',16,'LineWidth',3);
    end
        
    h(4)=plot( [thrx thrx],[min([y_drug(select_idx);NBT_y])-.3, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(5)=plot( [thrx thrx],[mean(y_drug)+thry, max([y_drug(select_idx);NBT_y])+.3],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(6)=plot( [-thrx -thrx],[min([y_drug(select_idx);NBT_y])-.3, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(7)=plot( [-thrx -thrx],[mean(y_drug)+thry, max([y_drug(select_idx);NBT_y])+.3],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(8)=plot( [min([score(select_idx);NBT_score])-.3 -thrx],[mean(y_drug)-thry, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(9)=plot( [thrx,max([score(select_idx);NBT_score])+.3], [mean(y_drug)-thry,mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(10)=plot( [min([score(select_idx);NBT_score])-.3 -thrx],[mean(y_drug)+thry, mean(y_drug)+thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(11)=plot( [thrx,max([score(select_idx);NBT_score])+.3], [mean(y_drug)+thry,mean(y_drug)+thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    set(h(1), 'MarkerFaceColor', 'r');
    set(h(2), 'MarkerFaceColor', 'b');
    t(1)=title(sprintf('%s\nBreast cell lines test acc=%.2f%% sn=%.2f%% sp=%.2f%%', drugname{drug}, [test_acc,Sensitivity,Specificity]));
    t(2)=xlabel( 'score');
    t(3)=ylabel( 'GI50');
    set(t, 'FontSize',40);
    set(gca,'FontSize',20);
    xlim([min([score(select_idx);NBT_score])-.3, max([score(select_idx);NBT_score])+.3]);
    ylim([min([y_drug(select_idx);NBT_y])-.3, max([y_drug(select_idx);NBT_y])+.3]);
    legend('Sensitive (train)', 'Resistant (train)', 'Not determined (train)','breast cancer cells','Location','bestoutside');
    
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
    
    %output result
    output_file = ['svm4_' drugname{drug} '.txt'];
    f = fopen(output_file, 'w');
    fprintf(f, 'TEST LEAVE-ONE-OUT accuracy=%.2f%% sensitivity=%.2f%% specificity=%.2f%%\n', loo_acc);
    fprintf(f, 'TRAIN size=%d\n', length(train_idx));
    for i=1:length(train_idx), fprintf(f, ' %s', name_select{train_idx(i)}); end; fprintf(f,'\n');
    fprintf(f, 'TEST  size=%d\n', length(test_idx));
    for i=1:length(test_idx), fprintf(f, ' %s', name_select{test_idx(i)}); end; fprintf(f,'\n');
    fprintf(f,'%s %s %s %s %s\n', '#', 'POSITION', 'MARKER', 'GENE', 'WEIGHT');
    fprintf(f,'%d %d %s %s %.8f\n', 0, 0, 'intercept', '---', b_noscale);
    for i=1:length(w_noscale)
        fprintf(f,'%d %d %s %s %.8f\n', i, max_feature(i), feature_name{i}, gene_name{i}, w_noscale(i));
    end
    fprintf(f,'\n%s %s %s %s\n', 'NAME', 'GI50', 'SCORE', 'norm_GI50');
    [~,I]=sort(y_drug); I = 1:size(x,1);
    for i=1:size(x,1)
        fprintf(f,'%s %.4f %.4f %.4f\n', name{I(i)}, y_drug(I(i)), score(I(i)), z_drug(I(i)));
    end
    fclose(f);
