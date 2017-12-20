import os

# os.system("pdflatex DTUNoter.tex")
# os.system("pdflatex DTUNoter.tex")

filenames = '.glo', '.idx', '.ilg', '.ind', '.ist', '.lof', '.lot', '.out', \
    '.log', '.aux', '.toc', 'bbl', '.gls', '.glg', '.blg'

filelist = [f for f in os.listdir(".") if f.endswith((filenames))]
for f in filelist:
    os.remove(f)


