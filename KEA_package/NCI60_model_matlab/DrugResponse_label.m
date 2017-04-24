function [label,y_drug,z_drug,std_y,name,x]=DrugResponse_label(drug,name,x,y,threshold,factor)
    %get label and remove non determined data
    y_drug = y(:,drug); 
    good_idx = ~isnan(y_drug);
    name = name(good_idx);
    x = x(good_idx,:);
    y_drug = y_drug(good_idx);
    mean_y = mean(y_drug);
    std_y = std(y_drug);
    z_drug = (y_drug-mean_y)/std_y;
    label = z_drug>0;
end