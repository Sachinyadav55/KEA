clear
close all
clc
%%
load drug_mas5_reduced
[~,n_drug] = size(drugname);
for drug = 1:n_drug
    input_file = ['svm_' drugname{drug}];
    load(input_file);
%%
    %print out leave one out result
    clf, hold on
    %h(1)=plot( lsa(tmp(pos_idx_l)), y_drug(pos_idx ) , 'dr','MarkerSize',24);
    %h(2)=plot( lsa(tmp(neg_idx_l)), y_drug(neg_idx ) , 'db','MarkerSize',24);
    %h(5)=plot( lsa(not_idx), y_drug(not_idx) , 'o','Color',[1,0.5,0], 'MarkerSize',20);
    
    h(1)=plot( lsa(pos_idx_l), y_drug(pos_idx ) , 'or' ,'MarkerSize',15,'LineWidth',3,'MarkerFaceColor', 'r');
    h(2)=plot( lsa(neg_idx_l), y_drug(neg_idx ) , 'ob' ,'MarkerSize',15,'LineWidth',3,'MarkerFaceColor', 'b');
    %h(3)=plot( score_t(~train_mark | not_idx), y_drug(~train_mark | not_idx) , 'dk','MarkerSize',18,'LineWidth',3);
    psx=score_t(~train_mark | not_idx);
    psy=y_drug(~train_mark | not_idx);
    psl=label(~train_mark | not_idx);
    lst=length(psx);
    for ki=1:lst
        if(psl(ki)>0)
            plot( psx(ki), psy(ki) , 'or' ,'MarkerSize',15,'LineWidth',3,'MarkerFaceColor', 'r');
        else
            plot( psx(ki), psy(ki) , 'ob' ,'MarkerSize',15,'LineWidth',3,'MarkerFaceColor', 'b');
        end
    end
    
    h(13)=plot( [min(lsa)-.3 max(lsa)+.3],[mean(y_drug)-thry, mean(y_drug)-thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    h(15)=plot( [min(lsa)-.3 max(lsa)+.3],[mean(y_drug)+thry, mean(y_drug)+thry],':', 'Color', [0.5 0.5 0.5],'LineWidth',3 );
    set(h(1), 'MarkerFaceColor', 'r');
    set(h(2), 'MarkerFaceColor', 'b');
    t(1)=title(sprintf('%s\nacc=%.2f%% sn=%.2f%% sp=%.2f%%', drugname{drug}, [test_acc,Sensitivity,Specificity]));
    t(2)=xlabel( 'score');
    t(3)=ylabel( 'GI50');
    set(t, 'FontSize',30);
    set(gca,'FontSize',30);
    xlim([min(lsa)-.3, max(lsa)+.3]);
    ylim([min(y_drug)-.3, max(y_drug)+.3]);
    legend('Sensitive','Resistant', 'Location','bestoutside');
    
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
    %%
    %fprintf(f,'\n%s %s %s %s\n', 'NAME', 'GI50', 'SCORE', 'norm_GI50');
    %[~,I_y]=sort(y_drug); I_y = 1:size(x,1);
    %for i=1:size(x,1)
    %    fprintf(f,'%s %.4f %.4f %.4f\n', name{I_y(i)}, y_drug(I_y(i)), lsa(I_y(i)), z_drug(I_y(i)));
    %end
    fclose(f);
end