# -*- coding: utf-8 -*-
"""
    @Author         : ASM TEAM
    @Purpose        : Test case apps***
    @Description    : Generic test cases configured to all applications
    @Date           : 05-08-2021
    @Last Modified  : 18-08-2021
"""
import os
import re
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

app_folder = os.path.dirname(os.path.realpath(__file__))
folder_list = [  # get all django apps on which these tests need to run
    f
    for f in os.listdir(BASE_DIR)
    if (
        os.path.isdir(os.path.join(BASE_DIR, f))
        and os.path.exists(os.path.join(BASE_DIR, f, "__init__.py"))
    )
]
module_yml_lookup = {
    "logging_hook": "cnfg_logging_hook.yml",
    "messaging": "cnfg_messaging.yml",
    "output_endpoint": "cnfg_output_endpoint.yml",
    "storage_spec": "cnfg_storage_sepc.yml",
}


def get_file_list(module):
    """get all the python scripts in a specifc module"""
    return [
        f
        for f in os.listdir(module)
        if os.path.isfile(os.path.join(module, f))
    ]


def get_function_list(module):
    """get all the functions names present in module"""
    function_list = []
    for file in get_file_list(module):
        if file != os.path.basename(__file__):
            content = open(os.path.join(module, file), "r")
            function_names = content.readlines()
            functions = [x for x in function_names if re.search("^def ", x)]
            if len(functions) > 0:
                for function in functions:
                    result = function.split("def ")[1].split("(")[0]
                    if result:
                        function_list.append(result)

    return function_list


def test_readme_exists():
    """test case to check the presence of README.md in the application"""
    error_list = []
    for module in folder_list:
        if not os.path.isfile(os.path.join(module, "README.md")):
            error_list.append(module)

    assert (
        error_list == []
    ), f"README.md file missing in the apps : {error_list}"


def test_function_details_in_readme():
    """test case to check the presence of all functions in README.md"""
    error_list = []
    readme_flag = False
    for module in folder_list:
        if "README.md" in get_file_list(module):
            readme_flag = True
            readme_looks_good = True
            file_obj = open(
                os.path.join(module, "README.md"), "r", encoding="utf-8"
            )
            content = file_obj.read()
            file_obj.close()
            for function in get_function_list(module):
                if function not in content:
                    readme_looks_good = False
                    error_list.append(module)

    if readme_flag:
        error_list = list(set(error_list))
        assert (
            error_list == []
        ), f"You have not described all the functions/class well in your README.md file in apps : {error_list}"


def test_readme_contents():
    """test case to check the readme length"""
    error_list = []
    readme_flag = False
    for module in folder_list:
        if "README.md" in get_file_list(module):
            readme_flag = True
            readme_words = [
                word
                for line in open(
                    os.path.join(module, "README.md"), "r", encoding="utf-8"
                )
                for word in line.split()
            ]
            if len(readme_words) < 100:
                error_list.append(module)

    if readme_flag:
        error_list = list(set(error_list))
        assert (
            error_list == []
        ), f"Make your README.md file interesting! Add atleast 100 words in apps : {error_list}"


def test_readme_file_for_formatting():
    """test case to check the readme file formatting"""
    error_list = []
    readme_flag = False
    for module in folder_list:
        if "README.md" in get_file_list(module):
            readme_flag = True
            file_obj = open(
                os.path.join(module, "README.md"), "r", encoding="utf-8"
            )
            content = file_obj.read()
            file_obj.close()
            print(content.count("#"))
            if content.count("#") < 4:
                error_list.append(module)
    if readme_flag:
        error_list = list(set(error_list))
        assert (
            error_list == []
        ), f"Add comments in your README.md file in apps : {error_list}"


def test_indentations():
    """test case to check the indentations"""
    error_dict = {}
    for module in folder_list:
        module_dict = {}
        this_folder = os.path.join(BASE_DIR, module)
        files_list = [
            f
            for f in os.listdir(this_folder)
            if os.path.isfile(os.path.join(this_folder, f))
            and f.endswith(".py")
        ]

        for file in files_list:
            error_list_lines = []
            line_number = 0
            content = open(os.path.join(this_folder, file), "r")
            function_names = content.readlines()
            for line in function_names:
                spaces = len(line) - len(line.lstrip())
                line_number += 1
                if spaces % 4 != 0 and line.strip():
                    error_list_lines.append(line_number)
            if error_list_lines != []:
                module_dict.update({file: error_list_lines})
        if module_dict != {}:
            error_dict.update({module: module_dict})

    assert (
        error_dict == {}
    ), f"Your code indentation does not follow PEP8 guidelines in the modules {error_dict}"


def test_function_name_had_cap_letter():
    """test case to check naming convention for the functions"""
    error_list = []
    for module in folder_list:
        for function in get_function_list(module):
            if not function.islower():
                error_list.append([function, module])
    assert (
        error_list == []
    ), f"You have used Capital letter(s) in your function names in apps as shown : {error_list}"
