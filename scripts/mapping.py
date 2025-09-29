import re
import os, sys
import subprocess
from collections import defaultdict
import pprint

def process_ptx_file(dir_name: str, ptx_filepath: str, instruction_map: defaultdict):
    try:
        with open(ptx_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Fail: could not find '{ptx_filepath}': {e}")
        return False

    instruction_regex = re.compile(r"^\s*(?:@\S+\s+)?([a-zA-Z0-9.:_]+)")

    instructions_in_file = set()
    
    current_kernel_name = None
    in_kernel_body = False
    brace_level = 0

    for line in content.splitlines():
        if not in_kernel_body and '.entry' in line:
            match = re.search(r'\.entry\s+([^\s(]+)', line)
            if match:
                current_kernel_name = match.group(1)
                brace_level = 0
                continue

        if not current_kernel_name:
            continue

        if '{' in line:
            if not in_kernel_body:
                in_kernel_body = True
            brace_level += line.count('{')

        if '}' in line:
            brace_level -= line.count('}')

        if in_kernel_body:
            match = instruction_regex.search(line.strip())
            if match:
                instruction = match.group(1)
                # filter: .reg, .param
                if ('.' in instruction or not instruction.endswith(':')) and not instruction.startswith('.') :

                    instructions_in_file.add(instruction)

        if in_kernel_body and brace_level == 0:
            in_kernel_body = False
            current_kernel_name = None

    for instruction in instructions_in_file:
        instruction_map[instruction].add(dir_name)
    
    return True


if __name__ == '__main__':
    instruction_to_dirs_map = defaultdict(set)

    if len(sys.argv) < 2:
        print("Usage: python mapping.py DIR_TO_PTXs")
        exit(-1)
    search_root = sys.argv[1]
    print(f"--- Start searching .ptx recursively in folder: '{os.path.abspath(search_root)}' ---")

    ptx_files_found = 0
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith('.ptx'):
                ptx_files_found += 1
                ptx_filepath = os.path.join(root, file)
                
                dir_name = os.path.basename(root)
                if not "blackwell" in dir_name:
                    continue

                if dir_name == '.' or dir_name == '':
                    dir_name = '_root_directory_'
                
                print(f"\n[Processing] {ptx_filepath} (archived in: '{dir_name}')")
                
                process_ptx_file(
                    dir_name=dir_name,
                    ptx_filepath=ptx_filepath,
                    instruction_map=instruction_to_dirs_map
                )

    if ptx_files_found == 0:
         print("\n--- Could not find .ptx files ---")
    else:
        print(f"\n\n")
        print("=" * 30)
        print(f" Search done, found {ptx_files_found} .ptx files")
        print("=" * 30)
        
        # sort in alphabet order
        sorted_map = dict(sorted(instruction_to_dirs_map.items()))
        pprint.pprint(sorted_map)
    
    with open(os.path.join(search_root,"results.txt"), "w", encoding='utf-8') as f:
            for ins, example_set in sorted_map.items():
                if len(example_set) == 0:
                    f.write(f"{ins},")
                    continue
                for example in example_set:
                    f.write(f"{ins}, {example}\n")
    f.close()
