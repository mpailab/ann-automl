#!/bin/bash

# Запутить автогенерацию rst-файлов и помесить их в _pre_source
# sphinx-apidoc -f -o _pre_source ../ann_automl/
#
# Скопировать нужные файлы из _pre_source в source
# cp _pre_source/ann_automl.core.rst source
#
# Преобразовать README.md в .rst-формат и размесить результат в source
# m2r ../README.md; mv ../README.rst source/.
#
# Сформировать документацию из rst-файлов (файлы в _pre_source игнорируются)
make clean; make html
