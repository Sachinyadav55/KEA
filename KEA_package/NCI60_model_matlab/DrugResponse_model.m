function DrugResponse_model(inputfile,drug,feature_left_fix,factor,threshold)
    load(inputfile)
    drugname{drug} %print drug name
    
    %get label and remove non determined data
    [label,y_drug,z_drug,std_y,name,x]=DrugResponse_label(drug,name,x,y,threshold,factor);
	
    %%
    %select data and separate training and testing data
    select_idx = abs(z_drug) > threshold;
    n_select_idx = find(abs(z_drug) < threshold);
    fprintf('#samples=%d\n', sum(select_idx));
    
	x_select = x(select_idx,:);
    x_n_select = x(n_select_idx,:);
    y_select = y_drug(select_idx);
    y_n_select = y_drug(n_select_idx);
    z_select = z_drug(select_idx);
    z_n_select = z_drug(n_select_idx);
    label_select = z_select > 0;
    label_n_select = z_n_select > 0;
    name_select = name(select_idx);
    pos_z = find(z_select > 0); 
    [~,pos_I] = sort( z_select(pos_z) );
    neg_z = find(z_select < 0); 
    [~,neg_I] = sort( -z_select(neg_z) );
    train_idx = [pos_z(pos_I(1:floor(length(pos_I)*factor))); neg_z(neg_I(1:floor(length(neg_I)*factor)))];
    test_idx = setdiff([1:length(z_select)]', train_idx);
    
    n_s=length(test_idx);
    n_ns=length(n_select_idx);

    if n_s<n_ns
        r_m=(n_ns-n_s+1):n_ns;
    else
        r_m=1:n_ns;
    end
    x=[x_select;x(n_select_idx(r_m),:)];
    label_t=[label_select(test_idx);label(n_select_idx(r_m))];
    label_t=label_t';
    label=[label_select;label(n_select_idx(r_m))];
    y_drug=[y_select;y_drug(n_select_idx(r_m))];
    z_drug=[z_select;z_drug(n_select_idx(r_m))];
    name=[name_select;name(n_select_idx(r_m))];
    %%     
    %recersive calculate weights for features, and remove features base on the weights    
    remain = 1:size(x,2);
    sorted_feature = [];

    n_feature = 0; step = 100;
    while n_feature < size(x,2)
        svm = svmtrain(x_select(train_idx,remain), label_select(train_idx),'kernel_function','linear','boxconstraint',1,'method','SMO');
        w = svm.Alpha'*svm.SupportVectors;
        if (length(remain) > step) 
            [~,I]=sort(abs(w)); 
            I = I(step:-1:1); % step =100
        else
            [~,I]=min(abs(w)); 
            I = I(1); % step = 1
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
        xtrain = x_select(train_idx,sorted_feature(1:f)); 
        ytrain = label_select(train_idx);
        xtest = x_select(test_idx,sorted_feature(1:f)); 
        ytest = label_select(test_idx);
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
    max_feature = sorted_feature(1:I); accuracy = acc(1,I)*100;
    feature_name = marker(max_feature);
    gene_name = genename(max_feature);
%%
    %Leave One Out
    lsa=[];
    cp = classperf(label,'Positive',1,'Negative',0);
    for i=1:size(x,1)
        loo_test_idx = false(size(x,1),1); loo_test_idx(i) = true;
        loo_train_idx = ~loo_test_idx;
        loo_xtrain = x(loo_train_idx,max_feature); 
        loo_ytrain=label(loo_train_idx);
        loo_xtest = x(loo_test_idx,max_feature);
        svm = svmtrain(loo_xtrain, loo_ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
        classOut = svmclassify(svm,loo_xtest);
        cp = classperf(cp,classOut,loo_test_idx);
        loow = svm.Alpha'*svm.SupportVectors;
        loow_noscale = loow .* svm.ScaleData.scaleFactor;
        loob_noscale = sum(loow .* svm.ScaleData.scaleFactor .* svm.ScaleData.shift) + svm.Bias;
        looscore = loo_xtest*loow_noscale' + loob_noscale; looscore = -looscore;
        lsa=[lsa,looscore];
    end
%%    
    %print out leave one out result
    X = x_select(:,max_feature); Y = label_select;
    xtrain = x_select(train_idx,max_feature); ytrain = label_select(train_idx);
    svm = svmtrain(xtrain, ytrain,'kernel_function','linear','boxconstraint',1,'method','SMO');
    w = svm.Alpha'*svm.SupportVectors;
    XX = bsxfun(@plus, X, svm.ScaleData.shift); XX = bsxfun(@times, XX, svm.ScaleData.scaleFactor);
    (XX*w'+svm.Bias)' < 0 == Y';
    w_noscale = w .* svm.ScaleData.scaleFactor;
    b_noscale = sum(w .* svm.ScaleData.scaleFactor .* svm.ScaleData.shift) + svm.Bias;
    (X*w_noscale'+b_noscale)' < 0 == Y';
    assert( all((X*w_noscale'+b_noscale < 0) == (XX*w'+svm.Bias < 0) ) )   
    score = x(:,max_feature)*w_noscale' + b_noscale; score = -score;
    pos_idx = z_drug > threshold; 
    neg_idx = z_drug < -threshold; 
    not_idx = ~pos_idx & ~neg_idx;
    tmp = 1:sum(select_idx);
    train_mark = false(size(x,1),1); 
    train_mark(tmp(train_idx)) = true;
    thry = 0.5*std_y;
    output_file = ['svm4_' drugname{drug}];
    %%
    score_t=[lsa(~not_idx), score(not_idx)'];
    p_label=score_t>0;
    tp=length(intersect(find(p_label>0),find(label>0)));
    tn=length(intersect(find(p_label<1),find(label<1)));
    fp=length(intersect(find(p_label>0),find(label<1)));
    fn=length(intersect(find(p_label<1),find(label>0)));
    Sensitivity=tp/(tp+fn)*100;
    Specificity=tn/(fp+tn)*100;
    label=label+0;
    p_label=p_label+0;
    test_acc=(1-nansum(abs(p_label-label'))/length(p_label))*100;
    %%

    clf, hold on
 
    h(1)=plot( score(pos_idx & train_mark), y_drug(pos_idx & train_mark) , 'or','MarkerSize',18,'LineWidth',3);
    h(2)=plot( score(neg_idx & train_mark), y_drug(neg_idx & train_mark) , 'ob','MarkerSize',18,'LineWidth',3 );
    h(3)=plot( score_t(~train_mark | not_idx), y_drug(~train_mark | not_idx) , 'dk','MarkerSize',18,'LineWidth',3);  
    h(4)=plot( [min(lsa)-.3 max(lsa)+.3],[mean(y_drug)-thry, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(5)=plot( [min(lsa)-.3 max(lsa)+.3],[mean(y_drug)+thry, mean(y_drug)+thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    set(h(1), 'MarkerFaceColor', 'r');
    set(h(2), 'MarkerFaceColor', 'b');
    t(1)=title(sprintf('%s\nacc=%.2f%% sn=%.2f%% sp=%.2f%%', drugname{drug}, [test_acc,Sensitivity,Specificity]));
    t(2)=xlabel( 'score');
    t(3)=ylabel( 'GI50');
    set(t, 'FontSize',30);
    set(gca,'FontSize',30);
    xlim([min(lsa)-.3, max(lsa)+.3]);
    ylim([min(y_drug)-.3, max(y_drug)+.3]);
    legend('Sensitive (Train)','Resistant (Train)', 'Test', 'Location','bestoutside');
    
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
    fprintf(f, 'TEST LEAVE-ONE-OUT accuracy=%.2f%% sensitivity=%.2f%% specificity=%.2f%%\n', [test_acc,Sensitivity,Specificity]);
    fprintf(f,'%s %s %s %s %s\n', '#', 'POSITION', 'MARKER', 'GENE', 'WEIGHT');
    fprintf(f,'%d %d %s %s %.8f\n', 0, 0, 'intercept', '---', b_noscale);
    for i=1:length(w_noscale)
        fprintf(f,'%d %d %s %s %.8f\n', i, max_feature(i), feature_name{i}, gene_name{i}, w_noscale(i));
    end
    fprintf(f,'\n%s %s %s %s\n', 'NAME', 'GI50', 'SCORE', 'norm_GI50');
    [~,I]=sort(y_drug); I = 1:size(x,1);
    for i=1:size(x,1)
        fprintf(f,'%s %.4f %.4f %.4f\n', name{I(i)}, y_drug(I(i)), lsa(I(i)), z_drug(I(i)));
    end
    fclose(f);
end