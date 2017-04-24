build:
	R CMD Rscript -e "library(knitr); knit('DrugResponse.Rnw')"
	latex DrugResponse.tex
	bibtex DrugResponse.aux
	latex DrugResponse.tex
	latex DrugResponse.tex
	dvipdf DrugResponse.dvi

clean:
	rm -rf *.tex *.dvi *.pdf

