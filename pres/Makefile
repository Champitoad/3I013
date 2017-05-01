NAME ?= pres

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex
	latexmk -pdflatex=lualatex -pdf --shell-escape $<

view: $(NAME).pdf
	zathura $<

clean:
	latexmk -CA
	@rm -vf *.{aux,log,nav,out,pdf,snm,toc}

.PHONY: view clean
