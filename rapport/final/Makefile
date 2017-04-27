NAME ?= rapport
TEMPL = template.latex

all: $(NAME).pdf

$(NAME).pdf: $(NAME).md $(TEMPL)
	pandoc -s -f markdown+definition_lists+auto_identifiers+fenced_code_attributes --toc -t latex --latex-engine=pdflatex --template=$(TEMPL) $< --filter pandoc-fignos --filter pandoc-eqnos -o $@

view: $(NAME).pdf
	zathura $^

clean:
	rm $(NAME).pdf

.PHONY: view clean
