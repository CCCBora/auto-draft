#!/usr/bin/env python
# This script is taken from: https://github.com/rekka/latex-flatten

# A simple script for flattening LaTeX files by inlining included files.
#
#   - Supports `\include` and `\input` commands.
#   - Automatically adds extension `.tex` if the file does not have an extension.
#   - Handles multiple include commands per line, comments.
#   - Does not flatten recursively.

import re
import sys

if len(sys.argv)==3:
    main_name = sys.argv[1]
    output_name = sys.argv[2]
else:
    sys.exit('USAGE: %s main.tex output.tex' %sys.argv[0])

main = open(main_name,'r')
output = open(output_name,'w')

for line in main.readlines():
    s = re.split('%', line, 2)
    tex = s[0]
    if len(s) > 1:
        comment = '%' + s[1]
    else:
        comment = ''

    chunks = re.split(r'\\(?:input|include)\{[^}]+\}', tex)

    if len(chunks) > 1:
        for (c, t) in zip(chunks, re.finditer(r'\\(input|include)\{([^}]+)\}', tex)):
            cmd_name = t.group(1)
            include_name = t.group(2)
            if '.' not in include_name: include_name = include_name + '.tex'
            if c.strip(): output.write(c + '\n')
            output.write('% BEGIN \\' + cmd_name + '{' + include_name + '}\n')
            include = open(include_name, 'r')
            output.write(include.read())
            include.close()
            output.write('% END \\' + cmd_name + '{' + include_name + '}\n')
        tail = chunks[-1] + comment
        if tail.strip(): output.write(tail)
    else:
        output.write(line)

output.close()
main.close()