.PHONY: clean

all: report.pdf

report.pdf: README.md
	pandoc README.md -N --template=mytemplate.tex --variable mainfont="Constantia" --variable sansfont="Corbel" --variable monofont="Consolas" --variable fontsize=12pt --latex-engine=xelatex --toc -o report.pdf

clean:
	rm report.pdf
