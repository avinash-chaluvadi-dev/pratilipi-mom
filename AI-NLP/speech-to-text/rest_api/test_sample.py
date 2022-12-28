import pytest
import os,re

functionList = []
this_folder = os.path.dirname(os.path.realpath(__file__))
files_list = [f for f in os.listdir(this_folder) if os.path.isfile(os.path.join(this_folder,f))]
for file in files_list:        
    if file != "test_sample.py":
        content = open(os.path.join(this_folder,file),"r")
        function_names = content.readlines()
        functions = [x for x in function_names if re.search("^def ",x)]
        if len(functions) > 0:
            for function in functions:
                print("function >>>>>>>>>>>>>>>>>>>", function)
                result = function.split('def ')[1].split("(")[0]
                if result:
                    functionList.append(result)


''' test case to check the presence of README.md in the application'''
def test_readme_exists():
    assert os.path.isfile("README.md"), "README.md file missing!"


''' test case to check the presence of all functions in README.md'''
def test_function_details_in_readme():    
    READMELOOKSGOOD = True
    f = open("README.md", "r", encoding="utf-8")
    content = f.read()
    f.close()
    for c in functionList:
        if c not in content:
            READMELOOKSGOOD = False
            pass
    assert READMELOOKSGOOD == True, "You have not described all the functions/class well in your README.md file"

    
''' test case to check the readme length'''
def test_readme_contents():
    readme_words=[word for line in open('README.md', 'r', encoding="utf-8") for word in line.split()]
    assert len(readme_words) >= 100, "Make your README.md file interesting! Add atleast 200 words"

''' test case to check the readme file formatting'''
def test_readme_file_for_formatting():
    f = open("README.md", "r", encoding="utf-8")
    content = f.read()
    f.close()
    assert content.count("#") >= 5

''' test case to check the indentations '''
def test_indentations():    
    this_folder = os.path.dirname(os.path.realpath(__file__))
    files_list = [f for f in os.listdir(this_folder) if os.path.isfile(os.path.join(this_folder,f))]

    for file in files_list:        
        line_number = 0
        content = open(os.path.join(this_folder,file),"r")
        function_names = content.readlines()
        for line in function_names:        
            spaces = len(line) - len(line.lstrip())
            line_number+= 1
            if spaces % 4 != 0 and line.strip():  
                assert spaces % 4 == 0, "Your code indentation does not follow PEP8 guidelines in line number "+str(line_number)+ " of module "+str(file)

''' test case to check naming convention for the functions '''
def test_function_name_had_cap_letter():
    for function in functionList:
        assert len(re.findall('([A-Z])', function[0])) == 0, "You have used Capital letter(s) in your function names"

