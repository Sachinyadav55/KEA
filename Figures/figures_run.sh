#!/bin/bash
# DrugResponse_sample_test_multi.sh: Example of piping DrugResponse package to test cancer samples
# Input: folder with multiple samples
# Usage:
# DrugResponse_sample_test_URL.sh path.to.folder

E_2_9C=70
E_all_cancer_related=71
E_probe_gene=72
E_density_plot=73
E_Response_rate=74

echo "Run r scripts to generate figures"

cd ./F1_bar_roc/2_9C/
if R -q -e "source('bar.r');source('roc.r')"; then
echo "bar plot and roc for 2_9C compare successful"
mv 2_9c_roc.pdf ../..
mv 2_9c_bar.pdf ../..
cd ../../
else
echo "Fail to process model compare bar plot and roc"
exit $E_2_9C
fi

cd ./F1_bar_roc/all_cancer_related/
if R -q -e "source('bar.r');source('roc.r')"; then
echo "bar plot and roc for all_cancer_related compare successful"
mv all_cancer_related_roc.pdf ../..
mv all_cancer_related_bar.pdf ../..
cd ../../
else
echo "Fail to process model compare bar plot and roc"
exit $E_all_cancer_related
fi

cd ./F1_bar_roc/probe_gene/
if R -q -e "source('bar.r');source('roc.r')"; then
echo "bar plot and roc for probe_gene compare successful"
mv probe_gene_roc.pdf ../..
mv probe_gene_bar.pdf ../..
cd ../../
else
echo "Fail to process model compare bar plot and roc"
exit $E_probe_gene
fi

cd ./F2c_R_prediction_GEO_datasets_density/
if R -q -e "source('density_plot.r')"; then
mv 3sets_drugs_score.pdf ../
cd ../
echo "density plot successful"
else
echo "Fail to process density plot"
exit $E_density_plot
fi

cd ./F2d_Response_rate_regression/
if R -q -e "source('rr.r')"; then
mv Response_rate.pdf ../
cd ../
echo "response rate plot successful"
else
echo "Fail to process response rate plot"
exit $E_Response_rate
fi

echo "done"
exit 0
