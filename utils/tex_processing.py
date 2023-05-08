import os

def replace_title(save_to_path, title):
    # Define input and output file names
    input_file_name = os.path.join(save_to_path, "template.tex")
    output_file_name = os.path.join(save_to_path , "main.tex")

    # Open the input file and read its content
    with open(input_file_name, 'r') as infile:
        content = infile.read()
    content = content.replace(r"\title{TITLE} ", f"\\title{{{title}}} ")

    # Open the output file and write the modified content
    with open(output_file_name, 'w') as outfile:
        outfile.write(content)


# return all string in \cite{...} \citet{...} or \citep{...}.

# check if citations are in bibtex.


# replace citations

# sometimes the output may include thebibliography and bibitem . remove all of it.



